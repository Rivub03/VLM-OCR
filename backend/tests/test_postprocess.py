from app.postprocess import deterministic_nid_fields, normalise_fields, parse_response


def test_bangla_nid_digits_are_normalised() -> None:
    values = deterministic_nid_fields("NID: ১২৩৪৫৬৭৮৯০", "nid_front")
    assert values["nid_no"] == "1234567890"


def test_json_response_separates_text_and_fields() -> None:
    text, fields, warnings = parse_response('{"text":"Mina Akter", "fields":{"nid_no":"1234567890"}}')
    assert text == "Mina Akter"
    assert fields == {"nid_no": "1234567890"}
    assert warnings == []


def test_regex_fields_do_not_replace_valid_model_field_with_empty_value() -> None:
    fields = normalise_fields({"name": "Mina"}, "Name: Mina", "nid_front")
    assert fields is not None
    assert fields["name"] == "Mina"


def test_native_html_ocr_is_converted_to_readable_text() -> None:
    text, fields, warnings = parse_response('<div data-label="Text"><p>Hello</p></div><div><p>World</p></div>')
    assert text == "Hello\nWorld"
    assert fields is None
    assert warnings == []


def test_repeated_native_ocr_output_is_trimmed_with_a_warning() -> None:
    content = "".join('<div><p>National ID Card</p></div>' for _ in range(4))
    text, fields, warnings = parse_response(content)
    assert text == "National ID Card\nNational ID Card"
    assert fields is None
    assert warnings


def test_layout_metadata_is_never_treated_as_ocr_or_nid_data() -> None:
    text, fields, warnings = parse_response('[{"label":"Text","bbox":"1 2 3 4","count":20}]')
    assert text == ""
    assert fields is None
    assert "layout metadata" in warnings[0]
