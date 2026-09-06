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
        max_ocr_equations_per_document: int = 50,
    ):
        self.enabled = enabled
        self.checkpoint_path = checkpoint_path
        self.max_equations_per_page = max_equations_per_page
        self.max_ocr_equations_per_document = max_ocr_equations_per_document
        self._remaining_ocr_equations = max_ocr_equations_per_document
        self._model: Any | None = None
        self._model_unavailable = False

    def extract_pages(self, file_bytes: bytes) -> list[PageMathExtraction]:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
        try:
            self.start_document()
            return [self.extract_page(page) for page in document]
        finally:
            document.close()

    def start_document(self) -> None:
        """Reset the bounded OCR budget before extracting one PDF."""
        self._remaining_ocr_equations = self.max_ocr_equations_per_document

    def extract_page(self, page: pymupdf.Page) -> PageMathExtraction:
        """Extract equation metadata from an already-open page."""
        blocks = [
            (pymupdf.Rect(x0, y0, x1, y1), text.strip())
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks", sort=True)
            if text.strip()
        ]
        equation_regions = self._equation_regions(blocks, page.rect.width)
        equations = [self._transcribe_or_flag(page, region) for region in equation_regions]
        return PageMathExtraction(equations=equations)

    def _equation_regions(
        self,
        blocks: list[tuple[pymupdf.Rect, str]],
        page_width: float,
    ) -> list[pymupdf.Rect]:
        """Group PDF glyph fragments that visually form one displayed equation.

        Many textbooks position integral limits, fractions, and superscripts as
        separate PDF text blocks. OCR must receive their combined visual region,
        not an individual fragment such as ``dt`` or ``∫``.
        """
        seed_rectangles = [
            rectangle
            for rectangle, text in blocks
            if self._looks_like_equation(text, rectangle, page_width)
        ]
        regions: list[pymupdf.Rect] = []
        for rectangle in seed_rectangles:
            for index, region in enumerate(regions):
                if self._same_display_region(rectangle, region):
                    regions[index] = region | rectangle
                    break
            else:
                regions.append(rectangle)

        expanded_regions = [self._expand_region(region, blocks) for region in regions]
        merged_regions: list[pymupdf.Rect] = []
        for region in expanded_regions:
            for index, merged in enumerate(merged_regions):
                if self._same_display_region(region, merged):
                    merged_regions[index] = merged | region
                    break
            else:
                merged_regions.append(region)
        return sorted(merged_regions, key=lambda region: (region.y0, region.x0))[: self.max_equations_per_page]

    @staticmethod
    def _same_display_region(first: pymupdf.Rect, second: pymupdf.Rect) -> bool:
        vertical_gap = max(first.y0 - second.y1, second.y0 - first.y1, 0)
        return vertical_gap <= 14 and first.x0 <= second.x1 + 36 and second.x0 <= first.x1 + 36

    @staticmethod
    def _expand_region(region: pymupdf.Rect, blocks: list[tuple[pymupdf.Rect, str]]) -> pymupdf.Rect:
        """Include nearby short glyph blocks while excluding normal prose."""
        expanded = pymupdf.Rect(region)
        changed = True
        while changed:
            changed = False
            for rectangle, text in blocks:
                if not LocalMathExtractor._is_formula_fragment(text):
                    continue
                vertical_gap = max(rectangle.y0 - expanded.y1, expanded.y0 - rectangle.y1, 0)
                if vertical_gap > 10 or rectangle.x0 > expanded.x1 + 48 or expanded.x0 > rectangle.x1 + 48:
                    continue
                combined = expanded | rectangle
                if combined != expanded:
                    expanded = combined
                    changed = True
        return expanded

    @staticmethod
    def _is_formula_fragment(text: str) -> bool:
        compact_text = re.sub(r"\s+", " ", text).strip()
        words = re.findall(r"[A-Za-z]{2,}", compact_text)
        return len(compact_text) <= 100 and (
            len(words) <= 3 or bool(LocalMathExtractor._math_characters.search(compact_text))
        )

    def _looks_like_equation(self, text: str, rectangle: pymupdf.Rect, page_width: float) -> bool:
        if len(text) < 3 or len(text) > 500:
            return False
        words = re.findall(r"[A-Za-z]{2,}", text)
        if len(words) > 16:
            return False
        is_centered = abs(rectangle.x0 + rectangle.width / 2 - page_width / 2) <= page_width * 0.2
        has_math_symbols = bool(self._math_characters.search(text))
        has_variable_equation = bool(self._variable_equation.search(text))
        is_compact_numeric_line = (
            is_centered
            and len(words) <= 3
            and "|" not in text
            and not re.match(r"^(?:chapter|theorem|example)\b", text, re.IGNORECASE)
            and bool(re.search(r"\d", text))
        )
        return has_math_symbols or has_variable_equation or is_compact_numeric_line

    def _transcribe_or_flag(self, page: pymupdf.Page, rectangle: pymupdf.Rect) -> dict[str, Any]:
        if self._remaining_ocr_equations <= 0:
            return self._source_only_equation(rectangle)
        model = self._get_model()
        if model is None:
            return self._source_only_equation(rectangle)
        self._remaining_ocr_equations -= 1
        cropped_image = self._crop_equation(page, rectangle)
        latex = self._transcribe(model, cropped_image)
        if latex and self._is_valid_latex(latex):
            return {
                "latex": latex,
                "status": "needs_verification",
                "confidence": 0.5,
                "bounding_box": [round(value, 2) for value in rectangle],
            }
        return self._source_only_equation(rectangle)

    @staticmethod
    def _source_only_equation(rectangle: pymupdf.Rect) -> dict[str, Any]:
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

    def _transcribe(self, model: Any, image: Image.Image) -> str:
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
