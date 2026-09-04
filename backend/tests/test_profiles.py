from app.profiles import make_payload, profile_for


def test_surya_and_dots_have_explicit_profiles() -> None:
    assert profile_for("datalab-to/surya-ocr-2").name == "surya-ocr-2"
    assert profile_for("dots-studio/dots.ocr").name == "dots-ocr"


def test_schema_prompt_keeps_the_requested_schema() -> None:
    payload = make_payload("datalab-to/surya-ocr-2", "data:image/png;base64,AA==", "schema", {"invoice_no": "string"})
    text = payload["messages"][0]["content"][0]["text"]
    assert "invoice_no" in text
    assert payload["temperature"] == 0.0

