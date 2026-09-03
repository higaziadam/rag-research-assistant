from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str = ""
    table: str = ""
    figure_caption: str = ""
    source: str = ""
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    equations: List[Dict[str, Any]] = field(default_factory=list)
    type: str = "text"

    def to_text_for_search(self) -> str:
        parts = [self.text.strip(), self.table.strip(), self.figure_caption.strip()]
        return "\n".join(p for p in parts if p).strip()


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    table: str = ""
    figure_caption: str = ""
    source: str = ""
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    equations: List[Dict[str, Any]] = field(default_factory=list)
