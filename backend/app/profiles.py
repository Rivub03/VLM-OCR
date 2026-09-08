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
    # NID cards must transcribe a Bengali address *and* ninety MRZ characters,
    # and the MRZ is last in reading order.  Budgeting these modes below the
    # plain-text budget truncates precisely the fields being extracted.
    nid_max_tokens: int = 2048


SURYA = ModelProfile("surya-ocr-2", ("datalab-to/surya-ocr-2",), 2048, 1024, "html", nid_max_tokens=2048)
CHANDRA = ModelProfile("chandra-ocr-2", ("datalab-to/chandra-ocr-2", "datalab-to/chandra"), 4096, 2048, "html", nid_max_tokens=2048)
DOTS = ModelProfile("dots-ocr", ("dots-studio/dots.ocr", "rednote-hilab/dots.ocr"), 4096, 2048, "json", nid_max_tokens=2048)
GENERIC = ModelProfile("openai-compatible-vlm", (), 1536, 768, "text", nid_max_tokens=1536)

# These strings are the models' documented task contracts. Surya distinguishes
# layout JSON from full-page HTML solely by this exact prompt.
SURYA_FULL_PAGE_PROMPT = "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
SURYA_BLOCK_PROMPT = "OCR this block image to HTML."
CHANDRA_OCR_PROMPT = "OCR this image to HTML."

# dots.ocr ships a fixed prompt dictionary and was fine-tuned against those
# exact strings.  They are reproduced verbatim, keyed by their upstream names.
#
# Resist "improving" the wording.  An earlier revision appended two sentences to
# prompt_ocr to stop blank Blood Group fields suppressing a card's transcription;
# for a model of this size, drifting off the trained contract is itself a cause
# of the empty and looping responses that change was trying to fix.  Blank
# fields are handled in postprocessing instead, where they cost nothing.
DOTS_PROMPT_OCR = "Extract the text content from this image."

DOTS_PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

DOTS_PROMPT_GROUNDING_OCR = (
    "Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n"
)

NID_FRONT_FIELDS = ("name", "dob", "nid_no")
NID_BACK_FIELDS = ("blood_group", "place_of_birth", "issue_date", "mrz_line1", "mrz_line2", "mrz_line3")


def profile_for(model_id: str) -> ModelProfile:
    normalized = model_id.lower()
    for profile in (SURYA, CHANDRA, DOTS):
        if any(normalized.startswith(prefix) for prefix in profile.model_prefixes):
            return profile
    return GENERIC


def max_tokens_for(profile: ModelProfile, mode: str) -> int:
    if mode.startswith("nid_"):
        return profile.nid_max_tokens
    return profile.text_max_tokens if mode == "text" else profile.structured_max_tokens


def _dots_structured_prompt(schema: dict[str, Any] | None) -> str:
    fields = schema or {}
    return (
        "Extract visible printed text from this document. Return one valid JSON object only, "
        "with `text` containing the complete reading-order transcription and `fields` matching "
        "this object exactly. Keep source-language values literal; use null when absent. Schema: "
        + json.dumps(fields, ensure_ascii=False)
    )


def _dots_prompt(mode: str, schema: dict[str, Any] | None, use_layout: bool) -> str:
    if mode == "schema":
        return _dots_structured_prompt(schema)
    if not use_layout:
        return DOTS_PROMPT_OCR
    # The layout task is what makes tables, formulas, figures and reading order
    # recoverable at all: prompt_ocr returns a flat string with no structure.
    #
    # It is the wrong task for an identity card, and measurably so. On the
    # benchmark's NID fronts it collapses the card into run-on lines, degrades
    # character accuracy, and on dense cards falls into a repetition loop that
    # runs to the token limit and truncates. Cards are served by the plain OCR
    # task instead; see `nid_layout_prompt_enabled` for how the caller chooses.
    return DOTS_PROMPT_LAYOUT_ALL_EN


def grounding_prompt(bbox: tuple[int, int, int, int]) -> str:
    """dots.ocr's region-OCR contract, for re-reading one failed field."""
    return DOTS_PROMPT_GROUNDING_OCR + json.dumps(list(bbox))


def make_payload(
    model_id: str,
    image_data_url: str,
    mode: str,
    schema: dict[str, Any] | None,
    *,
    use_layout_prompt: bool = True,
    prompt_override: str | None = None,
    max_tokens_override: int | None = None,
    repetition_penalty: float = 1.05,
    temperature: float = 0.0,
) -> dict[str, Any]:
    profile = profile_for(model_id)
    max_tokens = max_tokens_override or max_tokens_for(profile, mode)
    if prompt_override is not None:
        prompt = prompt_override
    elif profile is SURYA:
        # A NID is a single dense card rather than a multi-block document. The
        # block contract prevents Surya from returning layout-only JSON for a
        # card, while retaining one OCR request for its complete image.
        prompt = SURYA_BLOCK_PROMPT if mode.startswith("nid_") else SURYA_FULL_PAGE_PROMPT
    elif profile is CHANDRA:
        prompt = CHANDRA_OCR_PROMPT
    elif profile is DOTS:
        prompt = _dots_prompt(mode, schema, use_layout_prompt)
    else:
        prompt = "Perform OCR on this printed document. Return all visible text in reading order as Markdown. Do not infer missing characters."
        if mode == "schema" and schema:
            prompt += " Return JSON only with a `text` field and a `fields` object matching: " + json.dumps(schema, ensure_ascii=False)
    return {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Greedy decoding on a degraded card photograph can fall into a repeat
        # loop that burns the whole token budget; on the benchmark those pages
        # score materially worse than clean ones. The penalty is the lever
        # against that, tuned on the train split.
        "repetition_penalty": repetition_penalty,
        "seed": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
    }
