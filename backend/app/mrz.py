"""ICAO 9303 TD1 machine-readable zone parsing, verification, and repair.

Bangladesh NID cards carry a TD1 MRZ: three lines of exactly thirty characters
drawn from ``[A-Z0-9<]``.  Three of its check digits were verified by hand
against real cards before this module was written:

    I<BGD464633509<39<<<<<<<<<<<<<
    6401182M3301144BGD<<<<<<<<<<<2      birth 640118 -> 2, expiry 330114 -> 4
    HAQUE<<MD<IZAZUL<<<<<<<<<<<<<<      composite -> 2

    I<BGD733180807<67<<<<<<<<<<<<<
    9401172F3304086BGD<<<<<<<<<<<8      birth 940117 -> 2, expiry 330408 -> 6
    POLY<<RABEA<AKTER<<<<<<<<<<<<<      composite -> 8

The composite digit covers fifty characters spanning both of the first two
lines, so it verifies nearly the whole zone in one arithmetic test.  That makes
the MRZ self-checking: a transcription either satisfies the printed digits or it
does not, with no model judgement involved.

One deliberate exception.  The document-number check digit at line 1 position 15
is ``<`` filler on both sample cards and does **not** satisfy the standard
algorithm over positions 6-14.  Enforcing it would reject every valid card, so
this module does not check it.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal


TD1_LINE_LENGTH = 30
TD1_LINE_COUNT = 3
MRZ_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_WEIGHTS = (7, 3, 1)
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Glyph pairs an OCR engine genuinely confuses on this font. Repair only ever
# substitutes within this map, and only when the printed check digits single out
# one answer, so it corrects transcription noise rather than inventing data.
CONFUSIONS: dict[str, str] = {
    "0": "ODQ", "O": "0DQ", "D": "0O", "Q": "0O",
    "1": "IL", "I": "1L", "L": "1I",
    "2": "Z", "Z": "2",
    "5": "S", "S": "5",
    "6": "G", "G": "6C",
    "8": "B", "B": "8",
    "U": "V", "V": "U",
    "K": "X<", "X": "K",
    "C": "G<",
    "<": "KC",
}

Status = Literal["valid", "repaired", "unverified", "absent"]


@dataclass(frozen=True)
class MrzResult:
    lines: tuple[str, ...] = ()
    status: Status = "absent"
    checks: dict[str, bool] = field(default_factory=dict)
    repairs: int = 0
    document_number: str | None = None
    birth_date: str | None = None
    expiry_date: str | None = None
    sex: str | None = None
    surname: str | None = None
    given_names: str | None = None

    @property
    def verified(self) -> bool:
        return self.status in {"valid", "repaired"}

    @property
    def name(self) -> str | None:
        parts = [part for part in (self.given_names, self.surname) if part]
        return " ".join(parts) or None


def character_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - 55
    return 0


def check_digit(value: str) -> str:
    """ICAO 9303 7-3-1 weighted modulus 10 check digit."""
    total = sum(character_value(character) * _WEIGHTS[index % 3] for index, character in enumerate(value))
    return str(total % 10)


def _composite_source(line1: str, line2: str) -> str:
    """The fifty characters the TD1 composite check digit is computed over."""
    return line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]


def _structural_violations(lines: list[str]) -> list[tuple[int, int]]:
    """Positions holding a character their field cannot legally contain.

    A check digit alone cannot catch every substitution: ``G`` has value 16, so
    reading ``640118`` as ``G40118`` shifts the weighted sum by exactly 70 and
    the printed digit still matches.  Requiring numeric fields to be numeric
    closes that gap, and the violating position also tells the repair search
    exactly where to look.
    """
    violations: list[tuple[int, int]] = []
    for index_line, line in enumerate(lines):
        for index_char, character in enumerate(line):
            if character not in _allowed(index_line, index_char):
                violations.append((index_line, index_char))
    return violations


def _verify(lines: list[str]) -> dict[str, bool]:
    line1, line2 = lines[0], lines[1]
    return {
        # Line 1's document-number check digit is intentionally absent here.
        "birth_date": check_digit(line2[0:6]) == line2[6],
        "expiry_date": check_digit(line2[8:14]) == line2[14],
        "composite": check_digit(_composite_source(line1, line2)) == line2[29],
        "structure": not _structural_violations(lines),
    }


def is_mrz_candidate(line: str) -> bool:
    """Could this transcription line be one row of a machine-readable zone?"""
    compact = re.sub(r"\s+", "", line).upper()
    if not 24 <= len(compact) <= 36:
        return False
    if not re.fullmatch(rf"[{MRZ_ALPHABET}]+", compact):
        return False
    # A run of filler characters is what separates an MRZ row from an ordinary
    # uppercase heading of a similar length.
    return "<" in compact


def _normalise_line(compact: str, index: int) -> str:
    """Bring one row to exactly thirty characters without disturbing its fields.

    Padding position matters. Rows one and three end in filler, so length is
    corrected at the tail. Row two ends with the composite check digit, so a
    dropped character is restored inside the optional-data run just before it.
    """
    if len(compact) == TD1_LINE_LENGTH:
        return compact
    if index == 1:
        head, tail = compact[:-1], compact[-1:]
        if len(compact) < TD1_LINE_LENGTH:
            return head + "<" * (TD1_LINE_LENGTH - len(compact)) + tail
        trimmed = re.sub(r"<+$", "", head)
        surplus = len(head) - (TD1_LINE_LENGTH - 1)
        keep = max(len(trimmed), len(head) - surplus)
        return head[:keep].ljust(TD1_LINE_LENGTH - 1, "<") + tail
    if len(compact) < TD1_LINE_LENGTH:
        return compact.ljust(TD1_LINE_LENGTH, "<")
    trimmed = re.sub(r"<+$", "", compact)
    return (trimmed if len(trimmed) >= TD1_LINE_LENGTH else trimmed.ljust(TD1_LINE_LENGTH, "<"))[:TD1_LINE_LENGTH]


def find_mrz_lines(lines: list[str]) -> list[str] | None:
    """Return the last run of three consecutive MRZ-shaped rows."""
    flags = [is_mrz_candidate(line) for line in lines]
    for start in range(len(lines) - TD1_LINE_COUNT, -1, -1):
        if all(flags[start:start + TD1_LINE_COUNT]):
            return [
                _normalise_line(re.sub(r"\s+", "", lines[start + offset]).upper(), offset)
                for offset in range(TD1_LINE_COUNT)
            ]
    return None


def _allowed(index_line: int, index_char: int) -> str:
    """Characters legal at one MRZ position, by field type."""
    if index_line == 1:
        if index_char in range(0, 7) or index_char in range(8, 15) or index_char == 29:
            return "0123456789"
        if index_char == 7:
            return "MF<"
        if index_char in range(15, 18):
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return MRZ_ALPHABET
    if index_line == 2:
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
    if index_char in range(0, 5):
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
    return MRZ_ALPHABET


def _editable_positions(lines: list[str]) -> list[tuple[int, int, str]]:
    """Positions the check digits actually constrain, with their alternatives."""
    positions: list[tuple[int, int, str]] = []
    for index_line, index_char in [(0, i) for i in range(5, 30)] + [(1, i) for i in range(0, 30)]:
        current = lines[index_line][index_char]
        # Filler is never a substitution source. Cards print long runs of it, and
        # admitting every one of them as a candidate edit makes almost any repair
        # look ambiguous, which suppresses the corrections that do matter.
        if current == "<":
            continue
        allowed = _allowed(index_line, index_char)
        alternatives = "".join(
            candidate for candidate in CONFUSIONS.get(current, "")
            if candidate in allowed and candidate != current
        )
        if alternatives:
            positions.append((index_line, index_char, alternatives))
    return positions


def _substitute(lines: list[str], edits: tuple[tuple[int, int, str], ...]) -> list[str]:
    mutated = [list(line) for line in lines]
    for index_line, index_char, character in edits:
        mutated[index_line][index_char] = character
    return ["".join(line) for line in mutated]


def repair(lines: list[str], max_edits: int = 2) -> tuple[list[str], int] | None:
    """Correct OCR glyph confusions when the check digits single out one answer.

    Returns None when no substitution satisfies every check, and also when more
    than one distinct substitution does.  Ambiguity is reported as a failure
    rather than resolved by preference: an MRZ that could plausibly read two ways
    is not evidence for either.
    """
    positions = _editable_positions(lines)
    if not positions:
        return None
    solutions: set[tuple[str, ...]] = set()
    edits_used = 0

    # A structurally illegal character localises the error precisely, which the
    # composite digit alone cannot do: many different single edits shift a
    # modulus-10 sum by the same amount. Try those positions on their own first,
    # so a decidable repair is not discarded as ambiguous.
    suspect = set(_structural_violations(lines))
    if suspect:
        for index_line, index_char, alternatives in positions:
            if (index_line, index_char) not in suspect:
                continue
            for character in alternatives:
                candidate = _substitute(lines, ((index_line, index_char, character),))
                if all(_verify(candidate).values()):
                    solutions.add(tuple(candidate))
        if len(solutions) == 1:
            return list(next(iter(solutions))), 1
        solutions.clear()

    for index_line, index_char, alternatives in positions:
        for character in alternatives:
            candidate = _substitute(lines, ((index_line, index_char, character),))
            if all(_verify(candidate).values()):
                solutions.add(tuple(candidate))
    if solutions:
        edits_used = 1
    if not solutions and max_edits >= 2:
        for first in range(len(positions)):
            line_a, char_a, alternatives_a = positions[first]
            for second in range(first + 1, len(positions)):
                line_b, char_b, alternatives_b = positions[second]
                for character_a in alternatives_a:
                    for character_b in alternatives_b:
                        candidate = _substitute(lines, ((line_a, char_a, character_a), (line_b, char_b, character_b)))
                        if all(_verify(candidate).values()):
                            solutions.add(tuple(candidate))
        if solutions:
            edits_used = 2
    if len(solutions) != 1:
        return None
    return list(next(iter(solutions))), edits_used


def _format_date(yymmdd: str, *, birth: bool) -> str | None:
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    year, month, day = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    # A birth year cannot be in the future; an expiry year is not in the distant
    # past.  The two-digit year is disambiguated against today accordingly.
    current = date.today().year % 100
    century = (1900 if year > current else 2000) if birth else 2000
    return f"{day:02d} {_MONTHS[month - 1]} {century + year}"


def _split_name(line: str) -> tuple[str | None, str | None]:
    body = line.rstrip("<")
    if "<<" not in body:
        cleaned = re.sub(r"<+", " ", body).strip()
        return (cleaned or None, None)
    surname, _, given = body.partition("<<")
    surname = re.sub(r"<+", " ", surname).strip()
    given = re.sub(r"<+", " ", given).strip()
    return (surname or None, given or None)


def parse_mrz(lines: list[str]) -> MrzResult:
    """Locate, verify, and where possible repair the MRZ in a transcription."""
    located = find_mrz_lines(lines)
    if not located:
        return MrzResult()

    checks = _verify(located)
    status: Status = "unverified"
    repairs = 0
    if all(checks.values()):
        status = "valid"
    else:
        repaired = repair(located)
        if repaired:
            located, repairs = repaired
            checks = _verify(located)
            status = "repaired"

    line1, line2, line3 = located
    document_number = line1[5:14].rstrip("<") or None
    surname, given_names = _split_name(line3)
    return MrzResult(
        lines=tuple(located),
        status=status,
        checks=checks,
        repairs=repairs,
        document_number=document_number,
        birth_date=_format_date(line2[0:6], birth=True) if checks["birth_date"] else None,
        expiry_date=_format_date(line2[8:14], birth=False) if checks["expiry_date"] else None,
        sex=line2[7] if line2[7] in {"M", "F"} else None,
        surname=surname,
        given_names=given_names,
    )
