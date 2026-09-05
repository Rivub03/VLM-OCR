import json
import re
from html import unescape
from typing import Any


BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NID_FRONT_FIELDS = ("name", "dob", "nid_no")
NID_BACK_FIELDS = ("blood_group", "place_of_birth", "issue_date", "mrz_line1", "mrz_line2", "mrz_line3")


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

    Surya occasionally repeats an HTML/text block until its token limit on
    degraded card photographs. Keeping the first occurrence is more useful
    than returning thousands of duplicate lines to the caller.
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


def parse_response(value: str) -> tuple[str, dict[str, Any] | None, list[str]]:
    clean = _strip_fences(value)
    if re.search(r"<(?:div|p|table|h[1-6])\b", clean, flags=re.IGNORECASE):
        text, truncated = truncate_repeated_output(html_to_markdown(clean))
        warnings = ["The OCR decoder started repeating content; duplicate output was removed. A sharper, closer NID photo may improve accuracy."] if truncated else []
        return text, None, warnings
    try:
        decoded = json.loads(clean)
    except json.JSONDecodeError:
        return clean, None, []
    if isinstance(decoded, list) and all(isinstance(item, dict) and {"label", "bbox", "count"}.issubset(item) for item in decoded):
        return "", None, [
            "The model returned document-layout metadata rather than OCR text. No fields were extracted; retry this NID after updating the service or use the dots.ocr model profile."
        ]
    if not isinstance(decoded, dict):
        return clean, None, ["The model returned JSON in an unexpected shape."]
    text = str(decoded.get("text") or decoded.get("markdown") or decoded.get("content") or "")
    fields = decoded.get("fields")
    if fields is None:
        fields = {key: item for key, item in decoded.items() if key not in {"text", "markdown", "content"}}
    if not isinstance(fields, dict):
        return text or clean, None, ["The structured fields response was not an object."]
    return text or clean, fields, []


def _lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in text.translate(BN_TO_EN).splitlines() if line.strip()]


def _label_value(lines: list[str], label: str, validator) -> str | None:
    expression = re.compile(rf"^{label}\s*[:,-]?\s*(.*)$", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = expression.search(line)
        if not match:
            continue
        candidates = [match.group(1).strip()]
        if index + 1 < len(lines):
            candidates.append(lines[index + 1].strip())
        for candidate in candidates:
            if candidate and validator(candidate):
                return candidate
    return None


def _valid_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,80}", value)) and not re.fullmatch(r"(?:print|address|date of birth|nid no\.?)", value, re.IGNORECASE)


def _extract_date(value: str) -> str | None:
    match = re.search(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[A-Za-z]*\s+\d{4}\b", value, re.IGNORECASE)
    if not match:
        match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", value)
    return match.group(0) if match else None


def _extract_nid(value: str) -> str | None:
    match = re.search(r"(?<!\d)(?:\d[\s-]*){10,17}(?!\d)", value)
    if not match:
        return None
    candidate = re.sub(r"[\s-]", "", match.group())
    return candidate if len(candidate) in {10, 13, 17} else None


def _extract_mrz(lines: list[str]) -> list[str] | None:
    candidates = [re.sub(r"\s+", "", line) for line in lines]
    candidates = [line for line in candidates if re.fullmatch(r"[A-Z0-9<]{20,44}", line)]
    return candidates[-3:] if len(candidates) >= 3 else None


def extract_nid_fields(text: str, mode: str) -> tuple[dict[str, Any], list[str]]:
    """Extract only supported English NID fields with source-text evidence."""
    lines = _lines(text)
    keys = NID_FRONT_FIELDS if mode == "nid_front" else NID_BACK_FIELDS
    fields: dict[str, Any] = {key: None for key in keys}
    if mode == "nid_front":
        fields["name"] = _label_value(lines, r"name", _valid_name)
        fields["dob"] = _label_value(lines, r"(?:date\s*of\s*birth|dob)", lambda value: _extract_date(value) is not None)
        if fields["dob"]:
            fields["dob"] = _extract_date(fields["dob"])
        fields["nid_no"] = _label_value(lines, r"(?:nid\s*(?:no\.?|number)?)", lambda value: _extract_nid(value) is not None)
        if fields["nid_no"]:
            fields["nid_no"] = _extract_nid(fields["nid_no"])
    else:
        blood = _label_value(lines, r"(?:blood\s*group|bg)", lambda value: bool(re.fullmatch(r"(?:A|B|AB|O)\s*[+-]", value, re.IGNORECASE)))
        fields["blood_group"] = re.sub(r"\s+", "", blood).upper() if blood else None
        fields["place_of_birth"] = _label_value(lines, r"(?:place\s*of\s*birth|birth\s*place)", _valid_name)
        issue_date = _label_value(lines, r"(?:issue\s*date|date\s*of\s*issue)", lambda value: _extract_date(value) is not None)
        fields["issue_date"] = _extract_date(issue_date) if issue_date else None
        mrz = _extract_mrz(lines)
        if mrz:
            fields.update(dict(zip(("mrz_line1", "mrz_line2", "mrz_line3"), mrz, strict=True)))
    warnings = [f"{key} could not be validated from the OCR transcription and was returned as null." for key, value in fields.items() if value is None]
    return fields, warnings


def deterministic_nid_fields(text: str, mode: str) -> dict[str, Any]:
    """Compatibility wrapper for callers that only require fixed NID fields."""
    return extract_nid_fields(text, mode)[0]


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
