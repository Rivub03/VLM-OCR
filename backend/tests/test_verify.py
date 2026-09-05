"""CPU reconciliation of NID fields that the first pass could not validate.

The recogniser itself is stubbed here so these tests exercise the reconciliation
*rules* - what may replace a failed value, and on what evidence - rather than
the accuracy of an ONNX model, which belongs to the benchmark.
"""

import asyncio
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.nid import extract_nid
from app.preprocess import RenderedPage
from app.verify import Reconciler


GOOD_MRZ = [
    "I<BGD733180807<67<<<<<<<<<<<<<",
    "9401172F3304086BGD<<<<<<<<<<<8",
    "POLY<<RABEA<AKTER<<<<<<<<<<<<<",
]
BACK_LABELS = "Blood Group: O+ Place of Birth: DHAKA Issue Date: 09 Apr 2018"


def blank_page() -> RenderedPage:
    output = io.BytesIO()
    Image.fromarray(np.full((400, 640, 3), 250, np.uint8)).save(output, format="PNG")
    return RenderedPage(1, output.getvalue(), width=640, height=400)


def reconciler(lines: list[str], **overrides) -> Reconciler:
    options = {"nid_verify_enabled": True, "nid_verify_max_workers": 1, **overrides}
    settings = SimpleNamespace(**options)
    instance = Reconciler(settings)
    instance._load = lambda: object()          # the engine is present
    instance._read = lambda image: lines       # ...and returns this
    return instance


def run(instance: Reconciler, text: str, mode: str = "nid_back"):
    return asyncio.run(instance.reconcile(extract_nid(text, mode), blank_page(), mode))


def test_a_clean_first_pass_never_invokes_the_second_engine() -> None:
    calls: list[int] = []
    instance = reconciler([])
    instance._read = lambda image: calls.append(1) or []
    result = run(instance, BACK_LABELS + "\n" + "\n".join(GOOD_MRZ))
    assert not result.unresolved
    assert calls == []


def test_a_failed_mrz_is_recovered_when_the_check_digits_confirm_the_reread() -> None:
    result = run(reconciler(GOOD_MRZ), BACK_LABELS)
    assert result.mrz.status == "valid"
    assert result.fields["mrz_line1"] == GOOD_MRZ[0]
    assert result.confidence["mrz_line1"] >= 0.85
    assert result.evidence["mrz_line1"]["source"].startswith("rapidocr:mrz")
    assert not result.unresolved


def test_an_mrz_reread_that_still_fails_its_digits_is_not_accepted() -> None:
    broken = [GOOD_MRZ[0], GOOD_MRZ[1][:-1] + "7", GOOD_MRZ[2]]
    result = run(reconciler(broken), BACK_LABELS)
    assert set(result.unresolved) == {"mrz_line1", "mrz_line2", "mrz_line3"}
    assert all(evidence.get("source") != "rapidocr" for evidence in result.evidence.values())


def test_labelled_fields_are_recovered_only_through_their_own_validators() -> None:
    result = run(reconciler([BACK_LABELS]), "")
    assert result.fields["blood_group"] == "O+"
    assert result.fields["issue_date"] == "09 Apr 2018"
    assert all(result.confidence[key] == 0.70 for key in ("blood_group", "issue_date"))


def test_free_text_fields_are_never_supplied_by_the_second_engine() -> None:
    """Their validator only asks "does this look like letters".

    That cannot confirm a reading, so a second opinion would replace a null with
    a confidently wrong value. Measured: the CPU recogniser drops spaces in
    wide-set capitals, and "MADHURIBISWAS" passes a letters-only check.
    """
    result = run(reconciler([BACK_LABELS, "Name MADHURIBISWAS"]), "", mode="nid_back")
    assert result.fields["place_of_birth"] is None
    assert "place_of_birth" in result.unresolved

    front = run(reconciler(["Name MADHURIBISWAS"]), "", mode="nid_front")
    assert front.fields["name"] is None


def test_nothing_runs_when_only_free_text_fields_are_unresolved() -> None:
    calls: list[int] = []
    instance = reconciler([])
    instance._read = lambda image: calls.append(1) or []
    result = run(instance, "Date of Birth 10 Oct 1998\nNID No. 102 707 5694", mode="nid_front")
    assert result.unresolved == ["name"]
    assert calls == []


def test_a_second_reading_that_fails_validation_leaves_the_field_null() -> None:
    """Disagreement is not resolved by preference; unsupported stays null."""
    result = run(reconciler(["Blood Group: XY", "Issue Date: sometime in 2018"]), "")
    assert result.fields["blood_group"] is None
    assert result.fields["issue_date"] is None
    assert "blood_group" in result.unresolved


def test_recovered_fields_drop_their_stale_null_warnings() -> None:
    result = run(reconciler([BACK_LABELS]), "")
    assert not any("issue_date could not be validated" in warning for warning in result.warnings)
    assert any("CPU verification engine" in warning for warning in result.warnings)
    # A field that stayed null must keep saying so.
    assert any("mrz_line1 could not be validated" in warning for warning in result.warnings)


def test_reconciliation_can_be_disabled() -> None:
    result = run(reconciler([BACK_LABELS], nid_verify_enabled=False), "")
    assert result.fields["blood_group"] is None


def test_a_missing_engine_degrades_quietly_to_the_first_pass() -> None:
    instance = Reconciler(SimpleNamespace(nid_verify_enabled=True, nid_verify_max_workers=1))
    instance._load = lambda: None
    result = run(instance, BACK_LABELS)
    assert result.fields["place_of_birth"] == "DHAKA"
    assert "mrz_line1" in result.unresolved
