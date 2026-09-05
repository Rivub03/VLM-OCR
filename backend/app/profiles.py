import json
from dataclasses import dataclass
from typing import Any, Literal


OutputFormat = Literal["html", "text", "json"]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_prefixes: tuple[str, ...]
    text_max_tokens: int
    structured_max_tokens: int
    native_output: OutputFormat


SURYA = ModelProfile("surya-ocr-2", ("datalab-to/surya-ocr-2",), 2048, 1536, "html")
CHANDRA = ModelProfile("chandra-ocr-2", ("datalab-to/chandra-ocr-2", "datalab-to/chandra"), 4096, 2048, "html")
DOTS = ModelProfile("dots-ocr", ("dots-studio/dots.ocr", "rednote-hilab/dots.ocr"), 2048, 1536, "json")
GENERIC = ModelProfile("openai-compatible-vlm", (), 1536, 768, "text")

# These strings are the models' documented task contracts. Surya distinguishes
# layout JSON from full-page HTML solely by this exact prompt.
SURYA_FULL_PAGE_PROMPT = "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
CHANDRA_OCR_PROMPT = "OCR this image to HTML."
DOTS_TEXT_PROMPT = "Extract the text content from this image."


def profile_for(model_id: str) -> ModelProfile:
    normalized = model_id.lower()
    for profile in (SURYA, CHANDRA, DOTS):
        if any(normalized.startswith(prefix) for prefix in profile.model_prefixes):
            return profile
    return GENERIC


def _dots_structured_prompt(mode: str, schema: dict[str, Any] | None) -> str:
    if mode == "nid_front":
        fields: dict[str, Any] = {"name": None, "name_bn": None, "father_name": None, "mother_name": None, "dob": None, "nid_no": None}
    elif mode == "nid_back":
        fields = {"address_bn": None, "blood_group": None, "place_of_birth": None, "issue_date": None, "mrz_line1": None, "mrz_line2": None, "mrz_line3": None}
    else:
        fields = schema or {}
    return (
        "Extract visible printed text from this document. Return one valid JSON object only, "
        "with `text` containing the complete reading-order transcription and `fields` matching "
        "this object exactly. Keep source-language values literal; use null when absent. Schema: "
        + json.dumps(fields, ensure_ascii=False)
    )


def make_payload(model_id: str, image_data_url: str, mode: str, schema: dict[str, Any] | None) -> dict[str, Any]:
    profile = profile_for(model_id)
    max_tokens = profile.structured_max_tokens if mode != "text" else profile.text_max_tokens
    if profile is SURYA:
        prompt = SURYA_FULL_PAGE_PROMPT
    elif profile is CHANDRA:
        prompt = CHANDRA_OCR_PROMPT
    elif profile is DOTS:
        prompt = DOTS_TEXT_PROMPT if mode == "text" else _dots_structured_prompt(mode, schema)
    else:
        prompt = "Perform OCR on this printed document. Return all visible text in reading order as Markdown. Do not infer missing characters."
        if mode == "schema" and schema:
            prompt += " Return JSON only with a `text` field and a `fields` object matching: " + json.dumps(schema, ensure_ascii=False)
    return {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
    }
