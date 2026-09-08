"""CPU-side reconciliation for NID fields that failed local validation.

The GPU budget is spent entirely on one dots.ocr instance, so the second opinion
here is a small ONNX detector/recogniser (PP-OCR via RapidOCR) running on the
CPU.  It costs no VRAM and is invoked only for the fields that could not be
validated, so a clean card still takes exactly one model request.

Reconciliation never *chooses* between two readings on a hunch. A replacement is
accepted only when it passes the same validator the first reading failed - the
MRZ check digits, an NID length of 10/13/17, a parseable date. When neither
reading qualifies the field stays null.
"""

import asyncio
import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import Settings
from .mrz import parse_mrz
from .nid import (
    NidExtraction,
    VALIDATORS,
    _label_value,
    LABEL_PATTERNS,
    normalise_value,
)
from .preprocess import RenderedPage


logger = logging.getLogger(__name__)

CONFIDENCE_RECONCILED = 0.70
# Fields whose validator is strong enough to *confirm* a second reading, rather
# than merely find it plausible:
#
#   mrz_*           ICAO check digits
#   nid_no          exactly 10, 13, or 17 digits
#   dob/issue_date  parses as a real date
#   blood_group     one of eight literal values
#
# `name` and `place_of_birth` are excluded on purpose. Their validator only
# asks "does this look like letters", which cannot tell a good reading from a
# bad one. Measured on the NID front train split, admitting them made accuracy
# worse: the CPU recogniser drops spaces in wide-set capitals ("MADHURIBISWAS"),
# and every such string passes a letters-only check, so a null was replaced by a
# confidently wrong value.
RECONCILABLE_FIELDS = frozenset({
    "nid_no", "dob", "issue_date", "blood_group", "mrz_line1", "mrz_line2", "mrz_line3",
})
MRZ_BAND_TOP = 0.60
# Scales tried, in order, until the MRZ satisfies its check digits. Recognition
# of a 30-character line is sensitive to the resampling chain in ways no single
# constant survives across cameras and scanners, so rather than tuning one value
# the check digits are allowed to choose: a scale is accepted only when the
# printed digits confirm it. The first entry succeeds on a clean card, so the
# rest cost nothing in the common case. Tune this list on the real benchmark.
MRZ_BAND_UPSCALES = (2.0, 1.6, 1.25)
MRZ_BAND_BORDER_PX = 24
MRZ_FIELDS = ("mrz_line1", "mrz_line2", "mrz_line3")


class Reconciler:
    """Re-reads failed NID fields with an independent CPU OCR engine."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine: Any = None
        self._unavailable = False

    def _load(self) -> Any:
        if self._engine is not None or self._unavailable:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            # The dependency ships its own models inside the wheel, but an image
            # built before it was added must keep working, just without a second
            # opinion.
            logger.warning("rapidocr_onnxruntime is not installed; NID reconciliation is disabled.")
            self._unavailable = True
            return None
        try:
            # Left unbounded the ONNX runtime claims every core, which starves
            # the event loop serving other requests on the same box.
            self._engine = RapidOCR(
                intra_op_num_threads=max(1, self.settings.nid_verify_max_workers),
                inter_op_num_threads=1,
            )
        except Exception:
            logger.warning("RapidOCR failed to initialise; NID reconciliation is disabled.", exc_info=True)
            self._unavailable = True
        return self._engine

    def _read(self, image: np.ndarray) -> list[str]:
        """Run the CPU engine and rebuild reading-order lines from its boxes."""
        engine = self._load()
        if engine is None:
            return []
        try:
            detections, _ = engine(image)
        except Exception:
            logger.warning("RapidOCR failed on a crop; skipping reconciliation for it.", exc_info=True)
            return []
        if not detections:
            return []
        items = []
        for box, text, _score in detections:
            points = np.asarray(box, dtype=float)
            items.append((float(points[:, 1].mean()), float(points[:, 0].min()),
                          float(points[:, 1].max() - points[:, 1].min()), str(text)))
        if not items:
            return []
        tolerance = max(6.0, float(np.median([item[2] for item in items])) * 0.6)
        items.sort(key=lambda item: item[0])
        lines: list[str] = []
        row: list[tuple[float, str]] = []
        anchor = items[0][0]
        for center_y, left, _height, text in items:
            if abs(center_y - anchor) > tolerance and row:
                lines.append(" ".join(text for _, text in sorted(row)))
                row, anchor = [], center_y
            row.append((left, text))
        if row:
            lines.append(" ".join(text for _, text in sorted(row)))
        return lines

    def _read_mrz_band(self, band: np.ndarray, fallback: list[str]):
        """Re-read the MRZ band, letting the check digits pick the scale.

        A white margin is added first: glyphs sitting against the crop edge are
        clipped by the detector, which silently loses the leading character of
        line one.
        """
        best = parse_mrz(fallback) if fallback else parse_mrz([])
        for scale in MRZ_BAND_UPSCALES:
            image = band
            if scale > 1.01:
                image = np.asarray(Image.fromarray(band).resize(
                    (max(1, int(band.shape[1] * scale)), max(1, int(band.shape[0] * scale))),
                    Image.Resampling.LANCZOS,
                ))
            padded = cv2.copyMakeBorder(
                image, MRZ_BAND_BORDER_PX, MRZ_BAND_BORDER_PX, MRZ_BAND_BORDER_PX, MRZ_BAND_BORDER_PX,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
            candidate = parse_mrz(self._read(padded))
            if candidate.verified:
                return candidate
            if candidate.status != "absent" and best.status == "absent":
                best = candidate
        return best

    async def reconcile(self, extraction: NidExtraction, page: RenderedPage, mode: str) -> NidExtraction:
        if not self.settings.nid_verify_enabled:
            return extraction
        if not set(extraction.unresolved) & RECONCILABLE_FIELDS:
            return extraction
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._reconcile_sync, extraction, page, mode)
        except Exception:
            logger.warning("NID reconciliation failed; returning the first-pass extraction.", exc_info=True)
            return extraction

    def _reconcile_sync(self, extraction: NidExtraction, page: RenderedPage, mode: str) -> NidExtraction:
        if self._load() is None:
            return extraction
        try:
            with Image.open(io.BytesIO(page.content)) as source:
                full = np.asarray(source.convert("RGB"))
        except Exception:
            logger.warning("Could not decode the rendered page for reconciliation.", exc_info=True)
            return extraction

        pending = set(extraction.unresolved) & RECONCILABLE_FIELDS
        if not pending:
            return extraction
        needs_labels = bool(pending - set(MRZ_FIELDS))
        needs_mrz = bool(pending & set(MRZ_FIELDS))
        # Reading the whole page costs as much as reading the MRZ band, so it is
        # only worth doing when a labelled field actually needs it.
        lines = self._read(full) if needs_labels else []
        resolved: list[str] = []

        if needs_mrz and mode == "nid_back":
            # The MRZ is small, dense and always at the foot of the card. An
            # upscaled band gives the recogniser the resolution it needs without
            # re-reading the whole page a second time.
            band = full[int(full.shape[0] * MRZ_BAND_TOP):, :]
            if band.size:
                mrz = self._read_mrz_band(band, lines)
                if mrz.verified:
                    for offset, key in enumerate(MRZ_FIELDS):
                        extraction.fields[key] = mrz.lines[offset]
                        extraction.confidence[key] = 0.95 if mrz.status == "valid" else 0.85
                        extraction.evidence[key] = {"source": f"rapidocr:mrz:{mrz.status}", "checks": mrz.checks}
                        resolved.append(key)
                    extraction.mrz = mrz
                    extraction.warnings.append(
                        "The MRZ was re-read with the CPU verification engine and now satisfies its printed check digits."
                    )

        for key in sorted(pending - set(MRZ_FIELDS)):
            if key not in LABEL_PATTERNS:
                continue
            candidate = _label_value(lines, LABEL_PATTERNS[key], VALIDATORS[key], key=key)
            if not candidate:
                continue
            extraction.fields[key] = normalise_value(key, candidate)
            extraction.confidence[key] = CONFIDENCE_RECONCILED
            extraction.evidence[key] = {"source": "rapidocr", "line": candidate}
            resolved.append(key)

        if resolved:
            names = ", ".join(sorted(set(resolved)))
            extraction.warnings.append(
                f"These fields were recovered by the CPU verification engine after the first pass could not "
                f"validate them: {names}. Each still had to pass its own validator."
            )
            extraction.warnings = [
                warning for warning in extraction.warnings
                if not any(warning.startswith(f"{key} could not be validated") for key in resolved)
            ]
            extraction.unresolved = [key for key in extraction.unresolved if key not in set(resolved)]
        return extraction
