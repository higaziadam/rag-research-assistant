"""Local, verification-first extraction of mathematical regions from PDFs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageMathExtraction:
    text: str
    equations: list[dict[str, Any]]


class LocalMathExtractor:
    """Detect equation-like regions and optionally transcribe them with local Pix2Tex.

    Pix2Tex does not expose calibrated confidence scores. Its transcriptions are
    therefore deliberately marked as needing source verification and are never
    promoted to grounded answer content automatically.
    """

    _math_characters = re.compile(r"[=<>\u2264\u2265\u2248\u222b\u2211\u221a\u00b1\u00d7\u00f7^_{}]")
    _variable_equation = re.compile(r"\b[a-zA-Z]\s*=\s*[^=]+")

    def __init__(
        self,
        enabled: bool,
        checkpoint_path: Path,
        max_equations_per_page: int = 4,
    ):
        self.enabled = enabled
        self.checkpoint_path = checkpoint_path
        self.max_equations_per_page = max_equations_per_page
        self._model: Any | None = None
        self._model_unavailable = False

    def extract_pages(self, file_bytes: bytes) -> list[PageMathExtraction]:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
        try:
            return [self._extract_page(page) for page in document]
        finally:
            document.close()

    def _extract_page(self, page: pymupdf.Page) -> PageMathExtraction:
        page_text = page.get_text("text", sort=True).strip()
        equations = []
        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text, *_ = block
            candidate = text.strip()
            rectangle = pymupdf.Rect(x0, y0, x1, y1)
            if not self._looks_like_equation(candidate, rectangle, page.rect.width):
                continue
            equations.append(self._transcribe_or_flag(page, rectangle))
            if len(equations) >= self.max_equations_per_page:
                break
        return PageMathExtraction(text=page_text, equations=equations)

    def _looks_like_equation(self, text: str, rectangle: pymupdf.Rect, page_width: float) -> bool:
        if len(text) < 3 or len(text) > 500:
            return False
        words = re.findall(r"[A-Za-z]{2,}", text)
        is_centered = abs(rectangle.x0 + rectangle.width / 2 - page_width / 2) <= page_width * 0.2
        has_math_symbols = bool(self._math_characters.search(text))
        has_variable_equation = bool(self._variable_equation.search(text))
        is_compact_numeric_line = is_centered and len(words) <= 12 and bool(re.search(r"\d", text))
        return has_math_symbols or has_variable_equation or is_compact_numeric_line

    def _transcribe_or_flag(self, page: pymupdf.Page, rectangle: pymupdf.Rect) -> dict[str, Any]:
        cropped_image = self._crop_equation(page, rectangle)
        latex = self._transcribe(cropped_image)
        if latex and self._is_valid_latex(latex):
            return {
                "latex": latex,
                "status": "needs_verification",
                "confidence": 0.5,
                "bounding_box": [round(value, 2) for value in rectangle],
            }
        return {
            "latex": "",
            "status": "source_only",
            "confidence": 0.0,
            "bounding_box": [round(value, 2) for value in rectangle],
        }

    @staticmethod
    def _crop_equation(page: pymupdf.Page, rectangle: pymupdf.Rect) -> Image.Image:
        padded_rectangle = pymupdf.Rect(
            rectangle.x0 - 8,
            rectangle.y0 - 8,
            rectangle.x1 + 8,
            rectangle.y1 + 8,
        )
        padded_rectangle = padded_rectangle & page.rect
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=padded_rectangle, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    def _transcribe(self, image: Image.Image) -> str:
        model = self._get_model()
        if model is None:
            return ""
        try:
            return str(model(image)).strip()
        except Exception:
            logger.exception("Local math OCR failed; preserving the original PDF region instead.")
            return ""

    def _get_model(self) -> Any | None:
        if self._model_unavailable or not self.enabled:
            return None
        if not self.checkpoint_path.is_file():
            logger.warning("Local math OCR is enabled but no Pix2Tex checkpoint exists at %s.", self.checkpoint_path)
            self._model_unavailable = True
            return None
        try:
            from munch import Munch
            from pix2tex.cli import LatexOCR

            self._model = LatexOCR(
                Munch(
                    {
                        "config": "settings/config.yaml",
                        "checkpoint": str(self.checkpoint_path.resolve()),
                        "no_cuda": True,
                        "no_resize": False,
                    }
                )
            )
            return self._model
        except Exception:
            logger.exception("Could not load local Pix2Tex; mathematical regions will remain source-only.")
            self._model_unavailable = True
            return None

    @staticmethod
    def _is_valid_latex(latex: str) -> bool:
        if not latex or len(latex) > 1_000 or any(character in latex for character in "\x00\r\n"):
            return False
        opening_to_closing = {"{": "}", "(": ")", "[": "]"}
        stack = []
        for character in latex:
            if character in opening_to_closing:
                stack.append(opening_to_closing[character])
            elif character in opening_to_closing.values():
                if not stack or character != stack[-1]:
                    return False
                stack.pop()
        return not stack and bool(re.search(r"[A-Za-z0-9\\]", latex))
