from app.profiles import make_payload, profile_for


def test_surya_and_dots_have_explicit_profiles() -> None:
    assert profile_for("datalab-to/surya-ocr-2").name == "surya-ocr-2"
    assert profile_for("datalab-to/chandra-ocr-2").name == "chandra-ocr-2"
    assert profile_for("dots-studio/dots.ocr").name == "dots-ocr"


def test_schema_prompt_keeps_the_requested_schema() -> None:
    payload = make_payload("dots-studio/dots.ocr", "data:image/png;base64,AA==", "schema", {"invoice_no": "string"})
    text = payload["messages"][0]["content"][0]["text"]
    assert "invoice_no" in text
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 1536


def test_completion_tokens_leave_context_room_for_images() -> None:
    text_payload = make_payload("datalab-to/surya-ocr-2", "data:image/png;base64,AA==", "text", None)
    nid_payload = make_payload("datalab-to/surya-ocr-2", "data:image/png;base64,AA==", "nid_front", None)
    assert text_payload["max_tokens"] == 2048
    assert nid_payload["max_tokens"] == 1024


def test_native_html_models_use_documented_ocr_prompts() -> None:
    surya = make_payload("datalab-to/surya-ocr-2", "data:image/png;base64,AA==", "nid_front", None)
    chandra = make_payload("datalab-to/chandra-ocr-2", "data:image/png;base64,AA==", "text", None)
    assert surya["messages"][0]["content"][0]["text"].startswith("OCR this image to HTML. Each block")
    assert chandra["messages"][0]["content"][0]["text"] == "OCR this image to HTML."
