"""TD1 machine-readable zone verification and repair.

The fixtures are transcriptions of two real NID backs. Their check digits were
computed by hand before the implementation existed, so these tests pin the
arithmetic against known-good cards rather than against the code's own output.
"""

import pytest

from app.mrz import check_digit, find_mrz_lines, parse_mrz, repair


SAMPLE_A = [
    "I<BGD464633509<39<<<<<<<<<<<<<",
    "6401182M3301144BGD<<<<<<<<<<<2",
    "HAQUE<<MD<IZAZUL<<<<<<<<<<<<<<",
]
SAMPLE_B = [
    "I<BGD733180807<67<<<<<<<<<<<<<",
    "9401172F3304086BGD<<<<<<<<<<<8",
    "POLY<<RABEA<AKTER<<<<<<<<<<<<<",
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("640118", "2"), ("330114", "4"), ("940117", "2"), ("330408", "6")],
)
def test_check_digit_matches_the_printed_digits(value: str, expected: str) -> None:
    assert check_digit(value) == expected


@pytest.mark.parametrize("lines", [SAMPLE_A, SAMPLE_B])
def test_real_cards_verify_against_all_three_check_digits(lines: list[str]) -> None:
    result = parse_mrz(lines)
    assert result.status == "valid"
    assert result.checks == {"birth_date": True, "expiry_date": True, "composite": True, "structure": True}


def test_verified_mrz_yields_cross_checkable_identity_data() -> None:
    result = parse_mrz(SAMPLE_A)
    assert result.birth_date == "18 Jan 1964"
    assert result.expiry_date == "14 Jan 2033"
    assert result.sex == "M"
    assert result.surname == "HAQUE"
    assert result.given_names == "MD IZAZUL"
    assert result.document_number == "464633509"


def test_the_line1_document_check_digit_is_not_enforced() -> None:
    """It is `<` filler on real BD cards; enforcing it rejects every valid one."""
    assert SAMPLE_A[0][14] == "<"
    assert check_digit(SAMPLE_A[0][5:14]) != SAMPLE_A[0][14]
    assert parse_mrz(SAMPLE_A).status == "valid"


def test_spaced_transcription_is_normalised() -> None:
    """Cards print the MRZ with wide letter spacing; OCR reproduces it."""
    spaced = [" ".join(SAMPLE_B[0]), SAMPLE_B[1], SAMPLE_B[2]]
    assert parse_mrz(spaced).status == "valid"


def test_dropped_filler_is_restored_without_moving_the_check_digit() -> None:
    """Row two ends with the composite digit, so padding goes before it."""
    short = [SAMPLE_B[0][:-1], SAMPLE_B[1][:-2] + SAMPLE_B[1][-1], SAMPLE_B[2][:-3]]
    result = parse_mrz(short)
    assert result.status == "valid"
    assert result.lines == tuple(SAMPLE_B)


@pytest.mark.parametrize(
    ("description", "line_index", "position", "wrong"),
    [
        ("O read for 0 in the expiry field", 1, 10, "O"),
        ("O read for 0 in the document number", 0, 10, "O"),
        ("G read for 6 in the birth date", 1, 0, "G"),
    ],
)
def test_glyph_confusions_are_repaired_when_the_digits_decide(description, line_index, position, wrong) -> None:
    lines = list(SAMPLE_A if line_index == 1 and position == 0 else SAMPLE_B)
    original = lines[line_index]
    lines[line_index] = original[:position] + wrong + original[position + 1:]
    result = parse_mrz(lines)
    assert result.status == "repaired", description
    assert result.lines[line_index] == original


def test_a_genuine_digit_error_is_reported_not_invented() -> None:
    """No substitution in the confusion set explains it, so nothing is guessed."""
    lines = list(SAMPLE_B)
    lines[1] = lines[1][:-1] + "7"
    result = parse_mrz(lines)
    assert result.status == "unverified"
    assert result.checks["composite"] is False


def test_ambiguous_repairs_are_refused() -> None:
    """Two readings that both satisfy the digits are evidence for neither."""
    assert repair(["A" * 30, "B" * 30, "C" * 30]) is None


def test_a_letter_in_a_numeric_field_is_rejected_structurally() -> None:
    """G has value 16, so 6->G shifts the weighted sum by exactly 70.

    The printed check digit still matches; only a field-shape rule catches it.
    """
    lines = list(SAMPLE_A)
    lines[1] = "G" + lines[1][1:]
    assert check_digit(lines[1][0:6]) == lines[1][6]
    assert parse_mrz(lines).lines[1] == SAMPLE_A[1]


def test_non_mrz_text_is_absent_rather_than_misread() -> None:
    result = parse_mrz(["Place of Birth: JHENAIDAH", "Issue Date: 15 Jan 2018", "ABCD<INVALID"])
    assert result.status == "absent"
    assert result.lines == ()
    assert find_mrz_lines(["one", "two"]) is None
