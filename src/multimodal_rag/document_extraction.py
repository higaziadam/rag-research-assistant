"""Layout-aware, local extraction of searchable PDF evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pymupdf


BoundingBox = list[float]


@dataclass
class ExtractedElement:
    """One page-local unit of evidence that can become a retrieval chunk."""

    type: str
    text: str
    bounding_box: BoundingBox
    section: str = ""
    table: str = ""
    figure_caption: str = ""
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class ExtractedPage:
    page_number: int
    elements: list[ExtractedElement]
    quality_flags: list[str]


class StructuredPdfExtractor:
    """Extract ordered text, tables, and figure captions without leaving the machine."""

    _figure_caption_pattern = re.compile(r"^(?:figure|fig\.?)\s*\d+[.:]?", re.IGNORECASE)
    _table_caption_pattern = re.compile(r"^table\s*\d+[.:]?", re.IGNORECASE)
    _numbered_heading_pattern = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S+")

    def __init__(self, max_table_characters: int = 8_000):
        self.max_table_characters = max_table_characters

    def extract_pages(self, file_bytes: bytes) -> list[ExtractedPage]:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
        try:
            return [self._extract_page(page, page_number) for page_number, page in enumerate(document, start=1)]
        finally:
            document.close()

    def _extract_page(self, page: pymupdf.Page, page_number: int) -> ExtractedPage:
        blocks = self._text_blocks(page)
        table_elements = self._table_elements(page)
        image_regions = self._image_regions(page)
        caption_blocks = [block for block in blocks if self._figure_caption_pattern.match(block[4])]
        table_regions = [pymupdf.Rect(*element.bounding_box) for element in table_elements]
        elements: list[ExtractedElement] = [*table_elements]
        current_section = ""

        for x0, y0, x1, y1, text in blocks:
            rectangle = pymupdf.Rect(x0, y0, x1, y1)
            if self._overlaps_any(rectangle, table_regions):
                continue
            if self._figure_caption_pattern.match(text):
                elements.append(self._figure_element(text, rectangle, image_regions))
                continue
            if self._table_caption_pattern.match(text):
                self._attach_table_caption(elements, text, rectangle)
                continue
            if self._is_heading(text):
                current_section = text
                continue
            elements.append(
                ExtractedElement(
                    type="text",
                    text=text,
                    bounding_box=self._serialize_rectangle(rectangle),
                    section=current_section,
                )
            )

        caption_rectangles = [pymupdf.Rect(*block[:4]) for block in caption_blocks]
        for image_region in image_regions:
            if not self._overlaps_any(image_region, caption_rectangles):
                elements.append(
                    ExtractedElement(
                        type="figure",
                        text="Figure region detected without an extractable caption.",
                        figure_caption="Figure region detected without an extractable caption.",
                        bounding_box=self._serialize_rectangle(image_region),
                        quality_flags=["figure_caption_missing"],
                    )
                )

        page_flags = self._page_quality_flags(blocks, image_regions)
        for element in elements:
            element.quality_flags = sorted(set([*page_flags, *element.quality_flags]))
        return ExtractedPage(page_number=page_number, elements=elements, quality_flags=page_flags)

    @staticmethod
    def _text_blocks(page: pymupdf.Page) -> list[tuple[float, float, float, float, str]]:
        blocks = []
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks", sort=True):
            normalized_text = re.sub(r"\s+", " ", text).strip()
            if normalized_text:
                blocks.append((x0, y0, x1, y1, normalized_text))
        return blocks

    def _table_elements(self, page: pymupdf.Page) -> list[ExtractedElement]:
        try:
            tables = page.find_tables().tables
        except Exception:
            return []

        elements = []
        for table in tables:
            rows = table.extract()
            markdown = self._rows_to_markdown(rows)
            if not markdown:
                continue
            rectangle = pymupdf.Rect(table.bbox)
            elements.append(
                ExtractedElement(
                    type="table",
                    text=self._table_to_prose(rows),
                    table=markdown,
                    bounding_box=self._serialize_rectangle(rectangle),
                )
            )
        return elements

    def _figure_element(
        self,
        caption: str,
        caption_rectangle: pymupdf.Rect,
        image_regions: list[pymupdf.Rect],
    ) -> ExtractedElement:
        closest_image = self._closest_image(caption_rectangle, image_regions)
        bounding_box = self._serialize_rectangle(closest_image or caption_rectangle)
        quality_flags = [] if closest_image else ["figure_region_not_detected"]
        return ExtractedElement(
            type="figure",
            text=caption,
            figure_caption=caption,
            bounding_box=bounding_box,
            quality_flags=quality_flags,
        )

    @staticmethod
    def _image_regions(page: pymupdf.Page) -> list[pymupdf.Rect]:
        rectangles = []
        for image in page.get_images(full=True):
            rectangles.extend(page.get_image_rects(image[0]))
        return [rectangle for rectangle in rectangles if rectangle.width > 20 and rectangle.height > 20]

    @staticmethod
    def _closest_image(caption: pymupdf.Rect, image_regions: Iterable[pymupdf.Rect]) -> pymupdf.Rect | None:
        candidates = list(image_regions)
        if not candidates:
            return None
        return min(candidates, key=lambda region: min(abs(region.y0 - caption.y1), abs(region.y1 - caption.y0)))

    @staticmethod
    def _attach_table_caption(elements: list[ExtractedElement], caption: str, rectangle: pymupdf.Rect) -> None:
        tables = [element for element in elements if element.type == "table"]
        if not tables:
            return
        closest_table = min(
            tables,
            key=lambda element: abs(pymupdf.Rect(element.bounding_box).y0 - rectangle.y1),
        )
        closest_table.text = f"{caption} {closest_table.text}".strip()

    @staticmethod
    def _is_heading(text: str) -> bool:
        words = text.split()
        if len(words) > 14 or len(text) > 120 or text.endswith((".", ",", ";", ":")):
            return False
        uppercase_words = sum(word.isupper() for word in words if len(word) > 1)
        title_case_words = sum(word[:1].isupper() and word[1:].islower() for word in words if len(word) > 1)
        return (
            bool(StructuredPdfExtractor._numbered_heading_pattern.match(text))
            or uppercase_words >= max(2, len(words) - 1)
            or (len(words) <= 10 and title_case_words >= max(1, len(words) - 1))
        )

    @staticmethod
    def _overlaps_any(rectangle: pymupdf.Rect, regions: Iterable[pymupdf.Rect]) -> bool:
        for region in regions:
            intersection = rectangle & region
            if intersection.is_empty:
                continue
            if intersection.get_area() / max(rectangle.get_area(), 1) >= 0.35:
                return True
        return False

    def _rows_to_markdown(self, rows: list[list[str | None]]) -> str:
        cleaned_rows = [[self._clean_cell(cell) for cell in row] for row in rows if row]
        if not cleaned_rows:
            return ""
        width = max(len(row) for row in cleaned_rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in cleaned_rows]
        header = normalized_rows[0]
        body = normalized_rows[1:] or [[""] * width]
        lines = [f"| {' | '.join(header)} |", f"| {' | '.join(['---'] * width)} |"]
        lines.extend(f"| {' | '.join(row)} |" for row in body)
        return "\n".join(lines)[: self.max_table_characters]

    @staticmethod
    def _table_to_prose(rows: list[list[str | None]]) -> str:
        cleaned_rows = [[StructuredPdfExtractor._clean_cell(cell) for cell in row] for row in rows if row]
        if not cleaned_rows:
            return ""
        header = cleaned_rows[0]
        values = []
        for row in cleaned_rows[1:6]:
            pairs = [f"{header[index]}: {value}" for index, value in enumerate(row) if value and index < len(header)]
            if pairs:
                values.append("; ".join(pairs))
        return "Table: " + ". ".join(values or [" ".join(header)])

    @staticmethod
    def _clean_cell(cell: str | None) -> str:
        return re.sub(r"\s+", " ", cell or "").replace("|", "\\|").strip()

    @staticmethod
    def _page_quality_flags(
        blocks: list[tuple[float, float, float, float, str]],
        image_regions: list[pymupdf.Rect],
    ) -> list[str]:
        character_count = sum(len(block[4]) for block in blocks)
        if character_count == 0:
            return ["no_extractable_text"]
        if character_count < 80 and image_regions:
            return ["image_dominant_page"]
        if character_count < 80:
            return ["limited_extractable_text"]
        return []

    @staticmethod
    def _serialize_rectangle(rectangle: pymupdf.Rect) -> BoundingBox:
        return [round(value, 2) for value in (rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1)]


def render_pdf_region(pdf_path: Path, page_number: int, bounding_box: BoundingBox) -> bytes:
    """Render a cited PDF region as a PNG for the source viewer."""
    document = pymupdf.open(pdf_path)
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError("Page number is outside this PDF.")
        page = document[page_number - 1]
        rectangle = pymupdf.Rect(bounding_box) & page.rect
        if rectangle.is_empty or rectangle.width < 1 or rectangle.height < 1:
            raise ValueError("The cited region is outside this PDF page.")
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), clip=rectangle, alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()
