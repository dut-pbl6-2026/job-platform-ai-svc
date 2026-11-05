from dataclasses import dataclass, field


@dataclass
class ParsedCv:
    text: str
    parser: str  # text | ocr | mixed | docx
    pages: int
    pages_processed: int
    warnings: list[str] = field(default_factory=list)
    hidden_removed: int = 0
