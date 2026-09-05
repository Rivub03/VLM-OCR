from app.postprocess import deterministic_nid_fields, extract_nid_fields, normalise_fields, parse_response


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


def test_front_fields_are_label_aware_and_use_the_next_english_line() -> None:
    fields, warnings = extract_nid_fields("Name\nMST. KOHINUR BEGUM\nDate of Birth 28 Oct 1983\nNID No. 370 809 0620", "nid_front")
    assert fields == {"name": "MST. KOHINUR BEGUM", "dob": "28 Oct 1983", "nid_no": "3708090620"}
    assert warnings == []


def test_back_fields_keep_blank_blood_group_null_and_preserve_mrz() -> None:
    text = """Blood Group:
Place of Birth: JHENAIDAH
Issue Date: 15 Jan 2018
I<BGD464633509<39<<<<<<<<<
6401182M3301144BGD<<<<<<<<<2
HAQUE<<MD<IZAZUL<<<<<<<<<"""
    fields, warnings = extract_nid_fields(text, "nid_back")
    assert fields["blood_group"] is None
    assert fields["place_of_birth"] == "JHENAIDAH"
    assert fields["issue_date"] == "15 Jan 2018"
    assert fields["mrz_line1"] == "I<BGD464633509<39<<<<<<<<<"
    assert fields["mrz_line3"] == "HAQUE<<MD<IZAZUL<<<<<<<<<"
    assert any("blood_group" in warning for warning in warnings)


def test_blank_blood_group_on_a_shared_label_row_does_not_hide_other_back_fields() -> None:
    text = """Blood Group:  Place of Birth: CHANDPUR  Issue Date: 16 Jan 2018
I<BGD509688789<79<<<<<<<<<<
8402013F3301155BGD<<<<<<<<<<2
BEGUM<<RUZINA<<<<<<<<<<<<<<<<"""
    fields, warnings = extract_nid_fields(text, "nid_back")
    assert fields["blood_group"] is None
    assert fields["place_of_birth"] == "CHANDPUR"
    assert fields["issue_date"] == "16 Jan 2018"
    assert fields["mrz_line1"] == "I<BGD509688789<79<<<<<<<<<<"
    assert any("blood_group" in warning for warning in warnings)


def test_invalid_mrz_is_not_emitted() -> None:
    fields, warnings = extract_nid_fields("Place of Birth: JHENAIDAH\nABCD<INVALID", "nid_back")
    assert fields["mrz_line1"] is None
    assert any("mrz_line1" in warning for warning in warnings)
