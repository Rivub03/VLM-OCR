import json
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Any


BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NID_FRONT_FIELDS = ("name", "dob", "nid_no")
NID_BACK_FIELDS = ("blood_group", "place_of_birth", "issue_date", "mrz_line1", "mrz_line2", "mrz_line3")

# dots.ocr's layout task emits these categories. 'Picture' carries no text by
# contract, so a block of that category with an empty body is correct output
# rather than a failed read.
LAYOUT_CATEGORIES = frozenset({
    "Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
    "Picture", "Section-header", "Table", "Text", "Title",
})


@dataclass(frozen=True)
class LayoutBlock:
    category: str
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "text": self.text, "bbox": list(self.bbox) if self.bbox else None}


@dataclass
class ParsedResponse:
    text: str = ""
    fields: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    layout: list[LayoutBlock] | None = None
    #: Structured rendering, kept separate from `text`. The plain text flattens
    #: a table to pipe-separated rows for reading; the markdown keeps the HTML
    #: the model emitted, which is the part worth preserving for a document.
    markdown: str = ""

    def __iter__(self):
        """Preserve the original three-value unpacking used by callers/tests."""
        return iter((self.text, self.fields, self.warnings))


def _strip_fences(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json|markdown)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def html_to_markdown(value: str) -> str:
    """Convert constrained Surya/Chandra OCR HTML into readable text."""
    value = re.sub(r"<(?:br)\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</(?:div|p|h[1-6]|li|tr)\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</(?:td|th)\s*>", " | ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    value = unescape(value).replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip(" |") for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def truncate_repeated_output(value: str) -> tuple[str, bool]:
    """Cut a decoder loop without issuing another OCR request.

    A VLM occasionally repeats a block until its token limit on degraded card
    photographs. Keeping the first occurrence is more useful than returning
    thousands of duplicate lines to the caller.
    """
    lines = value.splitlines()
    seen: dict[str, int] = {}
    kept: list[str] = []
    for line in lines:
        key = re.sub(r"\s+", " ", line).strip()
        if key and seen.get(key, 0) >= 2:
            return "\n".join(kept).strip(), True
        if key:
            seen[key] = seen.get(key, 0) + 1
        kept.append(line)
    return value, False


REPETITION_WARNING = (
    "The OCR decoder started repeating content; duplicate output was removed. "
    "A sharper, closer NID photo may improve accuracy."
)


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _is_dots_layout(decoded: Any) -> bool:
    """Recognise dots.ocr's layout task output.

    Distinct from Surya's layout metadata, which carries a `count` and no text
    and must keep being rejected outright.
    """
    if not isinstance(decoded, list) or not decoded:
        return False
    typed = [item for item in decoded if isinstance(item, dict)]
    if len(typed) != len(decoded):
        return False
    return all("bbox" in item and ("category" in item or "text" in item) for item in typed) and any(
        str(item.get("category", "")) in LAYOUT_CATEGORIES for item in typed
    )


def _layout_blocks(decoded: list[dict[str, Any]]) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for item in decoded:
        category = str(item.get("category") or "Text")
        text = item.get("text")
        blocks.append(LayoutBlock(
            category=category,
            text="" if text is None else str(text),
            bbox=_coerce_bbox(item.get("bbox")),
        ))
    return blocks


def render_layout(blocks: list[LayoutBlock]) -> tuple[str, str]:
    """Render layout blocks to (plain text, markdown), preserving reading order."""
    plain: list[str] = []
    markdown: list[str] = []
    for block in blocks:
        body = block.text.strip()
        if block.category == "Picture" and not body:
            markdown.append("![figure](figure)")
            continue
        if not body:
            continue
        if block.category == "Table":
            plain.append(html_to_markdown(body))
            markdown.append(body)
        elif block.category == "Formula":
            plain.append(body)
            markdown.append(f"$$\n{body}\n$$")
        elif block.category == "Title":
            plain.append(body)
            markdown.append(f"# {body}")
        elif block.category == "Section-header":
            plain.append(body)
            markdown.append(f"## {body}")
        else:
            plain.append(body)
            markdown.append(body)
    return "\n".join(plain).strip(), "\n\n".join(markdown).strip()


_HTML_BLOCK = re.compile(r"<(table|figure|ul|ol|pre)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_CAPTION = re.compile(r"^(?:figure|fig\.?|table|chart|exhibit)\s*\d+\s*[:.—-]", re.IGNORECASE)


def blocks_from_markdown(value: str) -> list[LayoutBlock]:
    """Recover layout blocks from a Markdown/HTML transcription.

    Not every checkpoint answers the layout task with the JSON object its prompt
    describes; the served `dots-studio/dots.ocr` returns Markdown with embedded
    HTML tables instead. The structure is still there, so it is read back out of
    the text. These blocks carry no `bbox`, which is why the NID extractor keeps
    a text-based path: geometry is unavailable here.
    """
    blocks: list[LayoutBlock] = []

    def add_prose(chunk: str) -> None:
        for paragraph in re.split(r"\n\s*\n", chunk):
            body = paragraph.strip()
            if not body:
                continue
            heading = re.match(r"^(#{1,6})\s+(.*)$", body, flags=re.DOTALL)
            if heading:
                category = "Title" if len(heading.group(1)) == 1 else "Section-header"
                blocks.append(LayoutBlock(category, heading.group(2).strip()))
                continue
            formula = re.fullmatch(r"\$\$(.+?)\$\$", body, flags=re.DOTALL)
            if formula:
                blocks.append(LayoutBlock("Formula", formula.group(1).strip()))
                continue
            if body.startswith("!["):
                blocks.append(LayoutBlock("Picture", ""))
                continue
            if _CAPTION.match(body):
                blocks.append(LayoutBlock("Caption", body))
                continue
            if re.match(r"^(?:[-*+]\s|\d+[.)]\s)", body):
                blocks.append(LayoutBlock("List-item", body))
                continue
            blocks.append(LayoutBlock("Text", body))

    cursor = 0
    for match in _HTML_BLOCK.finditer(value):
        add_prose(value[cursor:match.start()])
        tag = match.group(1).lower()
        blocks.append(LayoutBlock("Table" if tag == "table" else "Text", match.group(0).strip()))
        cursor = match.end()
    add_prose(value[cursor:])
    return blocks


def parse_response(value: str) -> ParsedResponse:
    clean = _strip_fences(value)
    if re.search(r"<(?:div|p|table|h[1-6])\b", clean, flags=re.IGNORECASE) and not clean.startswith(("[", "{")):
        # Surya/Chandra emit whole-page HTML that must be flattened. dots.ocr
        # emits Markdown with HTML tables, where the markup is the useful part.
        page_html = re.search(r"<(?:div|p|h[1-6])\b", clean, flags=re.IGNORECASE)
        if page_html:
            text, truncated = truncate_repeated_output(html_to_markdown(clean))
            return ParsedResponse(text, None, [REPETITION_WARNING] if truncated else [], markdown=text)
        blocks = blocks_from_markdown(clean)
        text, _ = render_layout(blocks)
        text, truncated = truncate_repeated_output(text)
        return ParsedResponse(
            text, None, [REPETITION_WARNING] if truncated else [], layout=blocks, markdown=clean,
        )
    try:
        decoded = json.loads(clean)
    except json.JSONDecodeError:
        blocks = blocks_from_markdown(clean)
        text, _ = render_layout(blocks)
        text, truncated = truncate_repeated_output(text)
        return ParsedResponse(
            text, None, [REPETITION_WARNING] if truncated else [],
            layout=blocks or None, markdown=clean,
        )

    if isinstance(decoded, list) and all(
        isinstance(item, dict) and {"label", "bbox", "count"}.issubset(item) for item in decoded
    ):
        return ParsedResponse("", None, [
            "The model returned document-layout metadata rather than OCR text. No fields were extracted; retry this NID after updating the service or use the dots.ocr model profile."
        ])

    if _is_dots_layout(decoded):
        blocks = _layout_blocks(decoded)
        text, markdown = render_layout(blocks)
        text, truncated = truncate_repeated_output(text)
        warnings = [REPETITION_WARNING] if truncated else []
        return ParsedResponse(text, None, warnings, layout=blocks, markdown=markdown)

    if not isinstance(decoded, dict):
        text, _ = truncate_repeated_output(clean)
        return ParsedResponse(text, None, ["The model returned JSON in an unexpected shape."])

    text = str(decoded.get("text") or decoded.get("markdown") or decoded.get("content") or "")
    fields = decoded.get("fields")
    if fields is None:
        fields = {key: item for key, item in decoded.items() if key not in {"text", "markdown", "content"}}
    if not isinstance(fields, dict):
        return ParsedResponse(text or clean, None, ["The structured fields response was not an object."])
    return ParsedResponse(text or clean, fields, [])


def deterministic_schema_fields(text: str, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if not schema:
        return None
    extracted: dict[str, Any] = {key: None for key in schema}
    for key in schema:
        label = re.escape(key.replace("_", " "))
        match = re.search(rf"{label}\s*[:,-]?\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
    return extracted


def normalise_fields(fields: dict[str, Any] | None, text: str, mode: str, schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    extracted: dict[str, Any] = {}
    if mode == "schema" and not fields:
        extracted = deterministic_schema_fields(text, schema) or extracted
    if fields:
        extracted = {**fields, **{key: value for key, value in extracted.items() if value}}
    return extracted or None


# NID extraction moved to app.nid. These re-exports keep the historical import
# surface working for existing callers and regression tests.
from .nid import (  # noqa: E402
    deterministic_nid_fields,
    extract_nid_fields,
)

__all__ = [
    "BN_TO_EN", "NID_FRONT_FIELDS", "NID_BACK_FIELDS", "LayoutBlock", "ParsedResponse",
    "html_to_markdown", "truncate_repeated_output", "parse_response", "render_layout",
    "deterministic_schema_fields", "normalise_fields",
    "deterministic_nid_fields", "extract_nid_fields",
]
