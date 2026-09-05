import json
import re
from html import unescape
from typing import Any


BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


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


def parse_response(value: str) -> tuple[str, dict[str, Any] | None, list[str]]:
    clean = _strip_fences(value)
    if re.search(r"<(?:div|p|table|h[1-6])\b", clean, flags=re.IGNORECASE):
        return html_to_markdown(clean), None, []
    try:
        decoded = json.loads(clean)
    except json.JSONDecodeError:
        return clean, None, []
    if not isinstance(decoded, dict):
        return clean, None, ["The model returned JSON in an unexpected shape."]
    text = str(decoded.get("text") or decoded.get("markdown") or decoded.get("content") or "")
    fields = decoded.get("fields")
    if fields is None:
        fields = {key: item for key, item in decoded.items() if key not in {"text", "markdown", "content"}}
    if not isinstance(fields, dict):
        return text or clean, None, ["The structured fields response was not an object."]
    return text or clean, fields, []


def deterministic_nid_fields(text: str, mode: str) -> dict[str, Any]:
    normalized = text.translate(BN_TO_EN)
    fields: dict[str, Any] = {}
    nid_match = re.search(r"(?<!\d)(?:\d[\s-]*){10,17}(?!\d)", normalized)
    if nid_match:
        candidate = re.sub(r"[\s-]", "", nid_match.group())
        if len(candidate) in {10, 13, 17}:
            fields["nid_no"] = candidate
    dob_match = re.search(r"(?:date\s*of\s*birth|dob|জন্ম\s*তারিখ)\s*[:,-]?\s*([^\n]+)", normalized, re.IGNORECASE)
    if dob_match:
        fields["dob"] = dob_match.group(1).strip()
    blood_match = re.search(r"(?:blood\s*group|bg|রক্তের\s*গ্রুপ)\s*[:,-]?\s*(A|B|AB|O)\s*([+-])", normalized, re.IGNORECASE)
    if blood_match:
        fields["blood_group"] = blood_match.group(1).upper() + blood_match.group(2)
    if mode == "nid_front":
        for field, label in (("name", "name"), ("father_name", r"father'?s?\s+name"), ("mother_name", r"mother'?s?\s+name")):
            match = re.search(rf"{label}\s*[:,-]?\s*([^\n]+)", text, re.IGNORECASE)
            if match:
                fields[field] = match.group(1).strip()
    return fields


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
    extracted = deterministic_nid_fields(text, mode) if mode.startswith("nid_") else {}
    if mode == "schema" and not fields:
        extracted = deterministic_schema_fields(text, schema) or extracted
    if fields:
        extracted = {**fields, **{key: value for key, value in extracted.items() if value}}
    return extracted or None
