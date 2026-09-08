from app.profiles import (
    DOTS_PROMPT_LAYOUT_ALL_EN,
    DOTS_PROMPT_OCR,
    grounding_prompt,
    make_payload,
    max_tokens_for,
    profile_for,
)


def test_surya_and_dots_have_explicit_profiles() -> None:
    assert profile_for("datalab-to/surya-ocr-2").name == "surya-ocr-2"
    assert profile_for("datalab-to/chandra-ocr-2").name == "chandra-ocr-2"
    assert profile_for("dots-studio/dots.ocr").name == "dots-ocr"


def test_schema_prompt_keeps_the_requested_schema() -> None:
    payload = make_payload("dots-studio/dots.ocr", "data:image/png;base64,AA==", "schema", {"invoice_no": "string"})
    text = payload["messages"][0]["content"][0]["text"]
    assert "invoice_no" in text
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 2048


def test_dots_uses_the_layout_task_for_documents() -> None:
    """The layout task is the only one that reports tables, captions and order."""
    payload = make_payload("dots-studio/dots.ocr", "data:image/png;base64,AA==", "text", None)
    assert payload["messages"][0]["content"][0]["text"] == DOTS_PROMPT_LAYOUT_ALL_EN


def test_dots_prompts_are_the_verbatim_trained_contract() -> None:
    """Wording drift off dots.ocr's trained prompt set degrades its output."""
    assert DOTS_PROMPT_OCR == "Extract the text content from this image."
    assert DOTS_PROMPT_LAYOUT_ALL_EN.startswith("Please output the layout information from the PDF image")
    assert "'Table'" in DOTS_PROMPT_LAYOUT_ALL_EN and "LaTeX" in DOTS_PROMPT_LAYOUT_ALL_EN
    assert grounding_prompt((1, 2, 3, 4)).endswith("[1, 2, 3, 4]")


def test_disabling_the_layout_prompt_falls_back_to_plain_transcription() -> None:
    payload = make_payload(
        "dots-studio/dots.ocr", "data:image/png;base64,AA==", "nid_front", None, use_layout_prompt=False,
    )
    assert payload["messages"][0]["content"][0]["text"] == DOTS_PROMPT_OCR


def test_nid_modes_are_never_budgeted_below_plain_text() -> None:
    """An NID back must fit a Bengali address plus ninety MRZ characters.

    The MRZ is printed last, so a short budget truncates exactly the fields the
    mode exists to extract.
    """
    for model in ("dots-studio/dots.ocr", "datalab-to/surya-ocr-2", "datalab-to/chandra-ocr-2", "some/other-vlm"):
        profile = profile_for(model)
        assert max_tokens_for(profile, "nid_back") >= max_tokens_for(profile, "schema")
        assert max_tokens_for(profile, "nid_back") >= profile.structured_max_tokens


def test_payload_suppresses_decoder_loops_without_losing_determinism() -> None:
    payload = make_payload("dots-studio/dots.ocr", "data:image/png;base64,AA==", "text", None)
    assert payload["temperature"] == 0.0
    assert payload["repetition_penalty"] > 1.0


def test_native_html_models_use_documented_ocr_prompts() -> None:
    surya = make_payload("datalab-to/surya-ocr-2", "data:image/png;base64,AA==", "nid_front", None)
    surya_document = make_payload("datalab-to/surya-ocr-2", "data:image/png;base64,AA==", "text", None)
    chandra = make_payload("datalab-to/chandra-ocr-2", "data:image/png;base64,AA==", "text", None)
    assert surya["messages"][0]["content"][0]["text"] == "OCR this block image to HTML."
    assert surya_document["messages"][0]["content"][0]["text"].startswith("OCR this image to HTML. Each block")
    assert chandra["messages"][0]["content"][0]["text"] == "OCR this image to HTML."
