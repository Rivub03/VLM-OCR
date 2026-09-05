"""Strict, evidence-backed extraction of the supported English NID fields.

Two extraction strategies live here and they are tried in order:

1. **Spatial.** When the model returns layout blocks, a field's value is the
   block sitting to the right of its printed label, or directly beneath it.
   This is how the card is actually laid out, so it survives the interleaved
   Bengali lines, two-column reading order and Markdown decoration that defeat
   line adjacency.
2. **Textual.** The original line-oriented parser, unchanged in behaviour, for
   models and prompts that return a flat transcription.

Values are only ever *located and validated*; nothing here infers a field the
transcription does not support. An unsupported value is null with a warning.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .mrz import MrzResult, parse_mrz
from .postprocess_types import LayoutBlockLike


BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NID_FRONT_FIELDS = ("name", "dob", "nid_no")
NID_BACK_FIELDS = ("blood_group", "place_of_birth", "issue_date", "mrz_line1", "mrz_line2", "mrz_line3")

# Labels as the model actually transcribes them, not as they are printed.
#
# `NID No.` comes back as `NIC No.` on most benchmark cards - the D reads as a C
# at card resolution - and requiring a literal "NID" dropped the number
# entirely. The same applies to the digit/letter confusions in "NID" itself.
# Widening a *label* pattern is safe: the value still has to satisfy its own
# validator, so a mislabelled match cannot emit an invalid number.
#
# Bengali labels are included deliberately. The cards are bilingual and the
# model often transcribes only the Bengali label above an English value; using
# it to *locate* an ASCII value does not make this a Bengali extractor, and the
# validators are unchanged.
LABEL_PATTERNS: dict[str, str] = {
    "name": r"(?:(?<![A-Za-z])name|নাম)",
    "dob": r"(?:date\s*of\s*birth|dob|জন্ম\s*তারিখ)",
    "nid_no": r"(?:(?:n[il1]d|nic|njd|fid|nib)\s*(?:no\.?|number)?|আইডি\s*নম্বর)",
    "blood_group": r"(?:blood\s*group|bg|রক্তের\s*গ্রুপ)",
    "place_of_birth": r"(?:place\s*of\s*birth|birth\s*place|জন্মস্থান)",
    "issue_date": r"(?:issue\s*date|date\s*of\s*issue|প্রদানের\s*তারিখ)",
}

# A parent's name shares the `নাম` label, so those rows must not be mistaken for
# the cardholder's.
PARENT_LABEL = re.compile(r"(?:পিতা|মাতা|father|mother)", re.IGNORECASE)

# Any label that can follow a value on the same physical row. A run-on
# transcription is sliced at whichever comes next.
FOLLOWING_LABEL = re.compile(
    r"\b(?:blood\s*group|bg|place\s*of\s*birth|birth\s*place|issue\s*date|date\s*of\s*issue"
    r"|date\s*of\s*birth|dob|n[il1]d\s*no|nic\s*no|name)\s*[:,.-]?",
    re.IGNORECASE,
)

CONFIDENCE_SPATIAL = 0.90
CONFIDENCE_TEXTUAL = 0.75
# Layout position rather than a printed label: weaker evidence, marked as such.
CONFIDENCE_STRUCTURAL = 0.55
CONFIDENCE_MRZ_VALID = 1.0
CONFIDENCE_MRZ_REPAIRED = 0.85


@dataclass
class NidExtraction:
    fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    mrz: MrzResult = field(default_factory=MrzResult)
    #: Fields whose value could not be validated; the reconciliation step
    #: re-reads exactly these and nothing else.
    unresolved: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Shared validators
# --------------------------------------------------------------------------

# Strings that are never a name on their own. Honorifics matter here: when the
# model interleaves scripts, the Latin prefix of a name row can be just `MD` or
# `MST`, and emitting that as the name is worse than emitting nothing.
_NAME_LABEL_WORDS = re.compile(
    r"^(?:print|address|date\s*of\s*birth|dob|n[il1]d\s*no\.?|nic\s*no\.?|name|national\s*id\s*card"
    r"|government|bangladesh|blood\s*group|of\s+person|md|mst|most|mrs|ms|mis|mr)\.?$",
    re.IGNORECASE,
)


def english_name_prefix(value: str) -> str | None:
    """The leading English-name-shaped run of a candidate.

    A card transcription frequently runs a value together with whatever follows
    it on the same row, in either script: ``MD ALMAS নাম Date of Birth 11 Feb
    1983``. The English name is the leading Latin run, and it ends where another
    script or a digit begins. This locates a printed value; it does not repair
    or complete one.
    """
    match = re.match(r"[A-Za-z][A-Za-z .'-]*", value.strip())
    if not match:
        return None
    candidate = re.sub(r"\s+", " ", match.group()).strip(" .-'")
    # A trailing honorific-less fragment like "MD" alone is a label artefact.
    return candidate if len(candidate) >= 2 else None


def _valid_name(value: str) -> bool:
    candidate = english_name_prefix(value)
    return bool(candidate) and not _NAME_LABEL_WORDS.fullmatch(candidate)


def _extract_date(value: str) -> str | None:
    match = re.search(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[A-Za-z]*\s+\d{4}\b", value, re.IGNORECASE)
    if not match:
        match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", value)
    return match.group(0) if match else None


def _extract_nid(value: str) -> str | None:
    match = re.search(r"(?<!\d)(?:\d[\s-]*){10,17}(?!\d)", value)
    if not match:
        return None
    candidate = re.sub(r"[\s-]", "", match.group())
    return candidate if len(candidate) in {10, 13, 17} else None


def _valid_blood_group(value: str) -> bool:
    return bool(re.fullmatch(r"(?:A|B|AB|O)\s*[+-]", value, re.IGNORECASE))


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "name": _valid_name,
    "dob": lambda value: _extract_date(value) is not None,
    "nid_no": lambda value: _extract_nid(value) is not None,
    "blood_group": _valid_blood_group,
    "place_of_birth": _valid_name,
    "issue_date": lambda value: _extract_date(value) is not None,
}

NORMALISERS: dict[str, Callable[[str], Any]] = {
    "dob": _extract_date,
    "issue_date": _extract_date,
    "nid_no": _extract_nid,
    "blood_group": lambda value: re.sub(r"\s+", "", value).upper(),
    "name": english_name_prefix,
    "place_of_birth": english_name_prefix,
}


def normalise_value(key: str, value: str) -> Any:
    return NORMALISERS.get(key, lambda item: item)(value)


# --------------------------------------------------------------------------
# Textual extraction (retained fallback)
# --------------------------------------------------------------------------

def _lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in text.translate(BN_TO_EN).splitlines() if line.strip()]


def _clean_label_value(value: str) -> str:
    """Remove harmless OCR/Markdown decoration without changing source text."""
    return value.strip().strip("*`|# ")


def _label_value(lines: list[str], label: str, validator, *, key: str = "") -> str | None:
    """Return a validated value attached to a known label only.

    Labels are matched anywhere in a row, not just at its start. Cards are
    bilingual and the model routinely emits the Bengali label first (``নাম
    Name``), prefixes it with stray characters, or runs the whole card onto one
    line - all of which a start anchor rejects, taking the value with it.

    The value is whatever follows the label up to the next known label, on that
    row or the one after. Nothing unlabelled is ever searched for.
    """
    # `ঃ` is the visarga, which Bengali labels use where English uses a colon.
    expression = re.compile(rf"{label}\s*[:,.\-ঃ]?\s*(.*)$", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = expression.search(line)
        if not match:
            continue
        if key == "name" and PARENT_LABEL.search(line[:match.start()]):
            continue  # A parent's name shares this label.
        inline_value = match.group(1)
        next_label = FOLLOWING_LABEL.search(inline_value)
        if next_label:
            inline_value = inline_value[:next_label.start()]
        candidates = [_clean_label_value(inline_value)]
        for offset in (1, 2):
            if index + offset >= len(lines):
                break
            following = lines[index + offset]
            if key == "name" and PARENT_LABEL.search(following):
                # A bare `পিতা`/`মাতা` row is a stray label the model emitted out
                # of order, and the cardholder's name can still follow it. Only
                # a parent label that *carries* a value means the next value
                # belongs to the parent, so stop there and nowhere else.
                if english_name_prefix(re.sub(r"^.*?(?:পিতা|মাতা|father|mother)\s*[:ঃ]?\s*", "", following)):
                    break
                continue
            candidates.append(_clean_label_value(following))
        # Cards print the English name directly beneath the Bengali one, and the
        # model sometimes emits the `Name` label after the value it belongs to.
        if index > 0:
            preceding = lines[index - 1]
            if not (key == "name" and PARENT_LABEL.search(preceding)):
                candidates.append(_clean_label_value(preceding))
        for candidate in candidates:
            if candidate and validator(candidate):
                return candidate
    return None


_HEADER_NOISE = re.compile(
    r"(?:government|people|republic|bangladesh|national\s*id|card|date\s*of\s*birth|dob"
    r"|n[il1]d|nic|fid|blood|place\s*of\s*birth|issue|name|print|address)",
    re.IGNORECASE,
)


def structural_name(lines: list[str]) -> str | None:
    """Last-resort name recovery from the card's fixed layout.

    Used only when no label-anchored candidate validated. This is weaker
    evidence than a label, so it is gated by `NID_NAME_STRUCTURAL_FALLBACK` and
    constrained hard: the cardholder's English name is printed in capitals above
    the date of birth, so only an all-caps Latin row in that region qualifies,
    and the first one wins because a parent's name is always printed below it.

    It stays inside the "no guessing" rule in one specific sense: the value must
    still be present in the transcription and satisfy the same validator. It
    does relax *where* the value may come from, which is why it is measured
    separately and can be switched off.
    """
    limit = len(lines)
    for index, line in enumerate(lines):
        if re.search(r"date\s*of\s*birth|dob", line, re.IGNORECASE):
            limit = index
            break
    for line in lines[:limit]:
        candidate = _clean_label_value(line)
        if not candidate or _HEADER_NOISE.search(candidate):
            continue
        # Card names are printed in capitals; requiring that rejects most of the
        # model's prose and any stray lower-case artefact.
        letters = [character for character in candidate if character.isalpha()]
        if not letters or sum(1 for character in letters if character.isupper()) < len(letters):
            continue
        name = english_name_prefix(candidate)
        if name and _valid_name(name) and " " in name:
            return name
    return None


def _extract_mrz(lines: list[str]) -> list[str] | None:
    candidates = [re.sub(r"\s+", "", line) for line in lines]
    candidates = [line for line in candidates if re.fullmatch(r"[A-Z0-9<]{20,44}", line)]
    return candidates[-3:] if len(candidates) >= 3 else None


# --------------------------------------------------------------------------
# Spatial extraction
# --------------------------------------------------------------------------

def _vertical_overlap(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    top, bottom = max(first[1], second[1]), min(first[3], second[3])
    shortest = min(first[3] - first[1], second[3] - second[1])
    return (bottom - top) / shortest if shortest > 0 else 0.0


def _horizontal_overlap(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left, right = max(first[0], second[0]), min(first[2], second[2])
    narrowest = min(first[2] - first[0], second[2] - second[0])
    return (right - left) / narrowest if narrowest > 0 else 0.0


def _spatial_candidates(blocks: list[LayoutBlockLike], index: int) -> list[str]:
    """Values that could belong to the label in block `index`, nearest first.

    A printed form puts a value to the right of its label or on the line below
    it; both are collected and left for the field's validator to choose between.
    """
    label = blocks[index]
    if not label.bbox:
        return []
    label_box = tuple(label.bbox)
    label_height = max(1.0, label_box[3] - label_box[1])
    right: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []
    for other_index, other in enumerate(blocks):
        if other_index == index or not other.bbox or not other.text.strip():
            continue
        box = tuple(other.bbox)
        if box[0] >= label_box[2] - label_height * 0.25 and _vertical_overlap(label_box, box) > 0.5:
            right.append((box[0] - label_box[2], other.text.strip()))
        elif box[1] >= label_box[3] - label_height * 0.25 and _horizontal_overlap(label_box, box) > 0.3:
            gap = box[1] - label_box[3]
            if gap <= label_height * 2.5:
                below.append((gap, other.text.strip()))
    right.sort(key=lambda item: item[0])
    below.sort(key=lambda item: item[0])
    return [text for _, text in right] + [text for _, text in below]


def _spatial_field(blocks: list[LayoutBlockLike], key: str) -> tuple[str, Any] | None:
    """Locate one field by its printed label using block geometry."""
    pattern = re.compile(rf"(?<![A-Za-z]){LABEL_PATTERNS[key]}\s*[:,-]?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    validator = VALIDATORS[key]
    for index, block in enumerate(blocks):
        body = block.text.translate(BN_TO_EN)
        match = pattern.search(body)
        if not match:
            continue
        # A value printed on the same line as its label, inside one block.
        inline = _clean_label_value(match.group(1))
        if inline and validator(inline):
            return inline, block.bbox
        # A block that holds several lines: the value may be the next one.
        remainder = body[match.end():].splitlines()
        for line in remainder[:2]:
            candidate = _clean_label_value(line)
            if candidate and validator(candidate):
                return candidate, block.bbox
        for candidate_text in _spatial_candidates(blocks, index):
            for line in candidate_text.splitlines()[:2]:
                candidate = _clean_label_value(line.translate(BN_TO_EN))
                if candidate and validator(candidate):
                    return candidate, block.bbox
    return None


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def extract_nid(
    text: str,
    mode: str,
    layout: list[LayoutBlockLike] | None = None,
    *,
    structural_name_fallback: bool = False,
) -> NidExtraction:
    """Extract the fixed NID key set for one side, with evidence and confidence."""
    lines = _lines(text)
    blocks = [block for block in (layout or []) if block.bbox] if layout else []
    keys = NID_FRONT_FIELDS if mode == "nid_front" else NID_BACK_FIELDS
    result = NidExtraction(fields={key: None for key in keys})

    labelled = ("name", "dob", "nid_no") if mode == "nid_front" else ("blood_group", "place_of_birth", "issue_date")
    for key in labelled:
        found = _spatial_field(blocks, key) if blocks else None
        if found:
            raw, bbox = found
            result.fields[key] = normalise_value(key, raw)
            result.confidence[key] = CONFIDENCE_SPATIAL
            result.evidence[key] = {"source": "layout", "bbox": list(bbox) if bbox else None}
            continue
        raw = _label_value(lines, LABEL_PATTERNS[key], VALIDATORS[key], key=key)
        if raw:
            result.fields[key] = normalise_value(key, raw)
            result.confidence[key] = CONFIDENCE_TEXTUAL
            result.evidence[key] = {"source": "text", "line": raw}

    if mode == "nid_front" and structural_name_fallback and not result.fields["name"]:
        recovered = structural_name(lines)
        if recovered:
            result.fields["name"] = recovered
            result.confidence["name"] = CONFIDENCE_STRUCTURAL
            result.evidence["name"] = {"source": "structural", "line": recovered}

    if mode == "nid_back":
        mrz = parse_mrz(lines)
        result.mrz = mrz
        if mrz.verified:
            confidence = CONFIDENCE_MRZ_VALID if mrz.status == "valid" else CONFIDENCE_MRZ_REPAIRED
            for offset, key in enumerate(("mrz_line1", "mrz_line2", "mrz_line3")):
                result.fields[key] = mrz.lines[offset]
                result.confidence[key] = confidence
                result.evidence[key] = {"source": f"mrz:{mrz.status}", "checks": mrz.checks}
            if mrz.status == "repaired":
                result.warnings.append(
                    f"The MRZ failed its printed check digits and was corrected with {mrz.repairs} "
                    "character substitution(s) that the check digits determine uniquely."
                )
        else:
            # Falling back to the unverified lines keeps today's behaviour when
            # the checksums cannot be satisfied, but records that they were not.
            fallback = _extract_mrz(lines)
            if fallback:
                for offset, key in enumerate(("mrz_line1", "mrz_line2", "mrz_line3")):
                    result.fields[key] = fallback[offset]
                    result.confidence[key] = 0.4
                    result.evidence[key] = {"source": "mrz:unverified", "checks": mrz.checks}
            if mrz.status == "unverified":
                failed = ", ".join(sorted(name for name, ok in mrz.checks.items() if not ok))
                result.warnings.append(
                    f"The MRZ did not satisfy its printed check digits ({failed}) and could not be "
                    "corrected unambiguously. Treat these lines as unverified."
                )

    for key, value in result.fields.items():
        if value is None:
            result.warnings.append(f"{key} could not be validated from the OCR transcription and was returned as null.")
            result.unresolved.append(key)
            result.confidence[key] = 0.0
        elif result.confidence.get(key, 0.0) < 0.5:
            result.unresolved.append(key)
    return result


def extract_nid_fields(text: str, mode: str, layout: list[LayoutBlockLike] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Extract only supported English NID fields with source-text evidence."""
    extraction = extract_nid(text, mode, layout)
    return extraction.fields, extraction.warnings


def deterministic_nid_fields(text: str, mode: str) -> dict[str, Any]:
    """Compatibility wrapper for callers that only require fixed NID fields."""
    return extract_nid(text, mode).fields
