from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_prefixes: tuple[str, ...]
    base_prompt: str


SURYA = ModelProfile(
    name="surya-ocr-2",
    model_prefixes=("datalab-to/surya-ocr-2",),
    base_prompt="Transcribe this printed document faithfully. Preserve reading order, headings, lists and tables in Markdown. Do not invent or correct text.",
)
DOTS = ModelProfile(
    name="dots-ocr",
    model_prefixes=("dots-studio/dots.ocr", "rednote-hilab/dots.ocr"),
    base_prompt="Parse this printed document and return its complete text in reading order as Markdown. Preserve table structure and do not invent or correct text.",
)
GENERIC = ModelProfile(
    name="openai-compatible-vlm",
    model_prefixes=(),
    base_prompt="Perform OCR on this printed document. Return all visible text in reading order as Markdown. Do not infer missing characters.",
)


def profile_for(model_id: str) -> ModelProfile:
    normalized = model_id.lower()
    for profile in (SURYA, DOTS):
        if any(normalized.startswith(prefix) for prefix in profile.model_prefixes):
            return profile
    return GENERIC


def make_payload(model_id: str, image_data_url: str, mode: str, schema: dict[str, Any] | None) -> dict[str, Any]:
    profile = profile_for(model_id)
    prompt = profile.base_prompt
    if mode == "nid_front":
        prompt += " Extract this Bangladesh NID front as JSON with: name, name_bn, father_name, mother_name, dob, nid_no. Return JSON only."
    elif mode == "nid_back":
        prompt += " Extract this Bangladesh NID back as JSON with: address_bn, blood_group, place_of_birth, issue_date, mrz_line1, mrz_line2, mrz_line3. Return JSON only."
    elif mode == "schema" and schema:
        prompt += " Return JSON only with a `text` field containing the transcription and a `fields` object matching this schema. Keep values literal and use null when not visible. Schema: " + __import__("json").dumps(schema, ensure_ascii=False)

    return {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
    }

