"""dots.ocr layout-task output: parsing, rendering, and spatial field matching."""

import json

from app.nid import extract_nid
from app.postprocess import LayoutBlock, parse_response, render_layout


def block(category: str, text: str, bbox: list[float]) -> dict:
    return {"category": category, "text": text, "bbox": bbox}


def test_dots_layout_json_becomes_blocks_not_a_raw_json_string() -> None:
    """Previously this fell through and the JSON source became the transcription.

    Every line-anchored label regex then matched nothing and each field nulled.
    """
    payload = json.dumps([
        block("Title", "National ID Card", [10, 10, 300, 40]),
        block("Text", "Name", [10, 60, 60, 80]),
        block("Text", "MUSA MIA", [80, 60, 220, 80]),
    ])
    parsed = parse_response(payload)
    assert parsed.layout is not None
    assert [item.category for item in parsed.layout] == ["Title", "Text", "Text"]
    assert "MUSA MIA" in parsed.text
    assert not parsed.text.lstrip().startswith("[")


def test_tables_stay_html_and_formulas_become_latex_in_markdown() -> None:
    blocks = [
        LayoutBlock("Title", "Quarterly Report", (0, 0, 10, 10)),
        LayoutBlock("Table", "<table><tr><td>Q1</td><td>42</td></tr></table>", (0, 20, 10, 40)),
        LayoutBlock("Formula", "E = mc^2", (0, 50, 10, 60)),
        LayoutBlock("Picture", "", (0, 70, 10, 90)),
    ]
    text, markdown = render_layout(blocks)
    assert "# Quarterly Report" in markdown
    assert "<table><tr><td>Q1</td><td>42</td></tr></table>" in markdown
    assert "$$\nE = mc^2\n$$" in markdown
    assert "![figure](figure)" in markdown
    # Plain text keeps the table readable and drops the figure placeholder.
    assert "Q1 | 42" in text
    assert "figure" not in text


def test_surya_layout_metadata_is_still_rejected_outright() -> None:
    parsed = parse_response('[{"label":"Text","bbox":"1 2 3 4","count":20}]')
    assert parsed.text == ""
    assert parsed.layout is None
    assert "layout metadata" in parsed.warnings[0]


def test_value_to_the_right_of_its_label_is_matched_by_geometry() -> None:
    """Line adjacency fails here: a Bengali line sits between label and value."""
    layout = [
        LayoutBlock("Text", "নাম", (40, 100, 90, 130)),
        LayoutBlock("Text", "সবুরা বেগম", (140, 100, 320, 130)),
        LayoutBlock("Text", "Name", (40, 140, 95, 165)),
        LayoutBlock("Text", "SABURA BEGUM", (140, 140, 360, 165)),
        LayoutBlock("Text", "Date of Birth", (40, 260, 150, 285)),
        LayoutBlock("Text", "01 Jan 1982", (170, 260, 300, 285)),
        LayoutBlock("Text", "NID No.", (40, 300, 110, 325)),
        LayoutBlock("Text", "911 616 1184", (170, 300, 330, 325)),
    ]
    flat = "\n".join(item.text for item in layout)
    extraction = extract_nid(flat, "nid_front", layout)
    assert extraction.fields == {"name": "SABURA BEGUM", "dob": "01 Jan 1982", "nid_no": "9116161184"}
    assert extraction.evidence["name"]["source"] == "layout"
    assert extraction.confidence["name"] > extraction.confidence.get("missing", 0)


def test_value_below_its_label_is_matched_by_geometry() -> None:
    layout = [
        LayoutBlock("Text", "Name", (40, 100, 95, 125)),
        LayoutBlock("Text", "MUSA MIA", (40, 130, 210, 158)),
    ]
    extraction = extract_nid("Name\nMUSA MIA", "nid_front", layout)
    assert extraction.fields["name"] == "MUSA MIA"


def test_markdown_table_decoration_no_longer_hides_a_value() -> None:
    """A start anchor rejected the row and took the value with it."""
    extraction = extract_nid("| Name | MUSA MIA |\n| Date of Birth | 10 Oct 1998 |", "nid_front")
    assert extraction.fields["name"] == "MUSA MIA"
    assert extraction.fields["dob"] == "10 Oct 1998"


def test_layout_extraction_falls_back_to_text_when_boxes_are_absent() -> None:
    extraction = extract_nid(
        "Name\nMST. KOHINUR BEGUM\nDate of Birth 28 Oct 1983\nNID No. 370 809 0620", "nid_front", None,
    )
    assert extraction.fields["name"] == "MST. KOHINUR BEGUM"
    assert extraction.evidence["name"]["source"] == "text"


def test_unresolved_fields_are_listed_for_reconciliation() -> None:
    extraction = extract_nid("Name\nMUSA MIA", "nid_front")
    assert extraction.fields["name"] == "MUSA MIA"
    assert set(extraction.unresolved) == {"dob", "nid_no"}
    assert extraction.confidence["dob"] == 0.0


def test_repeated_plain_text_output_is_truncated_for_every_model() -> None:
    """The guard used to run only on the HTML branch, so dots loops slipped past."""
    parsed = parse_response("\n".join(["National ID Card"] * 6))
    assert parsed.text == "National ID Card\nNational ID Card"
    assert parsed.warnings
