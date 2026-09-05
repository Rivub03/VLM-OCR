"""Evaluate the service against an external, untracked NID benchmark folder.

Scoring is deliberately identical to the earlier two-model benchmark
(`ocr-benchmark/evaluation/`), so numbers from the two are directly comparable:

* names and dates are whitespace-collapsed and compared case-insensitively;
* NID numbers are compared as digits only;
* a field whose ground truth is blank is excluded from the accuracy
  denominator and scored only for false positives.

`near_miss` is a diagnostic, never part of the headline number. It classifies a
*failed* comparison so a punctuation convention in the ground truth can be told
apart from a genuine misread.
"""

import argparse
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx


FIELD_MAP = {
    "nid_front": {"name": "name_en", "dob": "dob", "nid_no": "nid_number"},
    "nid_back": {
        "blood_group": "blood_group",
        "place_of_birth": "place_of_birth",
        "issue_date": "issue_date",
        "mrz_line1": "mrz_line1",
        "mrz_line2": "mrz_line2",
        "mrz_line3": "mrz_line3",
    },
}
CASE_INSENSITIVE = {"name", "place_of_birth", "dob", "issue_date"}


def normalise(value: Any, field: str) -> str:
    value = " ".join(str(value or "").strip().split())
    if field == "nid_no":
        return "".join(character for character in value if character.isdigit())
    return value.casefold() if field in CASE_INSENSITIVE else value


def classify_near_miss(expected: str, actual: str, field: str) -> str | None:
    """Why a comparison failed, for triage only. Never counted as a match."""
    if not expected:
        return None
    if not actual:
        return "missing"
    expected_norm, actual_norm = normalise(expected, field), normalise(actual, field)
    if expected_norm == actual_norm:
        return None
    depunct = lambda value: re.sub(r"[^\w\s]", "", value)  # noqa: E731
    if depunct(expected_norm) == depunct(actual_norm):
        return "punctuation_only"
    if depunct(expected_norm).replace(" ", "") == depunct(actual_norm).replace(" ", ""):
        return "spacing_only"
    if expected.strip() == actual.strip():
        return "case_only"
    expected_bare, actual_bare = depunct(expected_norm), depunct(actual_norm)
    honorifics = re.compile(r"^(?:md|mst|mrs|mr|ms|mohammad|muhammad)\s+")
    if honorifics.sub("", expected_bare) == honorifics.sub("", actual_bare):
        return "honorific_only"
    if sum(1 for a, b in zip(expected_bare, actual_bare) if a != b) <= 1 and abs(len(expected_bare) - len(actual_bare)) <= 1:
        return "one_character"
    return "different"


def _tally(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def documents_from_ground_truth(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(documents, dict) or not documents:
        raise ValueError(f"{path} must contain a non-empty {{'documents': {{...}}}} ground-truth mapping.")
    return documents


def image_for(image_dir: Path, image_id: str) -> Path:
    candidate = image_dir / image_id
    if candidate.exists():
        return candidate
    matches = list(image_dir.glob(f"{Path(image_id).stem}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not locate exactly one image for {image_id} in {image_dir}")
    return matches[0]


def request_page(client: httpx.Client, args, image: Path) -> dict[str, Any]:
    with image.open("rb") as source:
        response = client.post(
            f"{args.base_url.rstrip('/')}/api/v1/ocr",
            headers={"X-API-Key": args.api_key},
            files={"file": (image.name, source, mimetypes.guess_type(image.name)[0] or "application/octet-stream")},
            data={"mode": args.type},
        )
    response.raise_for_status()
    return response.json()["result"][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--type", choices=FIELD_MAP, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.getenv("OCR_API_KEY"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N documents (smoke runs).")
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Parallel requests. Keep at or below the backend's MAX_INFERENCE_CONCURRENCY.",
    )
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Provide --api-key or OCR_API_KEY.")

    documents = documents_from_ground_truth(args.benchmark_root / "data" / "ground_truth" / f"{args.type}_{args.split}.json")
    if args.limit:
        documents = dict(list(documents.items())[:args.limit])
    image_dir = args.benchmark_root / "data" / "images" / args.type / args.split

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    with httpx.Client(timeout=600) as client:
        def evaluate(item: tuple[str, Any]) -> tuple[str, Any, dict[str, Any] | None, str | None]:
            image_id, document = item
            try:
                return image_id, document, request_page(client, args, image_for(image_dir, image_id)), None
            except Exception as exc:  # A failed request is data, not a crash.
                return image_id, document, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            for index, (image_id, document, page, error) in enumerate(pool.map(evaluate, documents.items()), 1):
                if index % 25 == 0 or index == len(documents):
                    rate = index / max(1e-9, time.perf_counter() - started)
                    print(f"  {index}/{len(documents)} ({rate:.2f}/s)", flush=True)
                page = page or {}
                fields = page.get("fields") or {}
                confidence = page.get("field_confidence") or {}
                evidence = page.get("field_evidence") or {}
                truth = document.get("fields", {})
                for predicted_key, truth_key in FIELD_MAP[args.type].items():
                    expected = truth.get(truth_key)
                    actual = fields.get(predicted_key)
                    matched = (
                        normalise(actual, predicted_key) == normalise(expected, predicted_key)
                        and bool(normalise(expected, predicted_key))
                    )
                    rows.append({
                        "image_id": image_id,
                        "field": predicted_key,
                        "expected": expected,
                        "actual": actual,
                        "matched": matched,
                        "false_positive": not normalise(expected, predicted_key) and bool(normalise(actual, predicted_key)),
                        "null": actual is None,
                        # Which stage produced the value, so a regression can be
                        # attributed to prompting, geometry, or reconciliation.
                        "confidence": confidence.get(predicted_key),
                        "source": str((evidence.get(predicted_key) or {}).get("source") or ""),
                        "near_miss": None if matched else classify_near_miss(expected, actual, predicted_key),
                    })
                diagnostics.append({
                    "image_id": image_id,
                    "error": error,
                    "finish_reason": page.get("finish_reason"),
                    "layout_blocks": len(page.get("layout") or []),
                    "mrz_status": next(
                        (str((evidence.get(key) or {}).get("source", "")).split(":")[-1]
                         for key in ("mrz_line1",) if evidence.get(key)),
                        None,
                    ),
                    "warnings": page.get("warnings") or [],
                })

    per_field = {}
    for field in FIELD_MAP[args.type]:
        values = [row for row in rows if row["field"] == field and normalise(row["expected"], field)]
        scores = [row["confidence"] for row in values if row["confidence"] is not None]
        per_field[field] = {
            "total": len(values),
            "correct": sum(row["matched"] for row in values),
            "exact_match_rate": sum(row["matched"] for row in values) / len(values) if values else None,
            "null_rate": sum(row["null"] for row in values) / len(values) if values else None,
            "false_positives": sum(row["false_positive"] for row in rows if row["field"] == field),
            "mean_confidence": sum(scores) / len(scores) if scores else None,
            # Attribution: layout geometry, flat text, MRZ check digits, or the
            # CPU reconciliation pass.
            "sources": _tally(row["source"].split(":")[0] for row in values if row["source"]),
            "near_misses": _tally(row["near_miss"] for row in values if row["near_miss"]),
        }
    supported = [row for row in rows if normalise(row["expected"], row["field"])]
    blank = [row for row in rows if not normalise(row["expected"], row["field"])]
    overall = {
        "supported_field_total": len(supported),
        "supported_field_exact_matches": sum(row["matched"] for row in supported),
        "supported_field_exact_match_rate": sum(row["matched"] for row in supported) / len(supported) if supported else None,
        "blank_field_total": len(blank),
        "blank_field_false_positives": sum(row["false_positive"] for row in blank),
        # Pipeline health, separate from field accuracy. A non-zero truncation
        # count means max_tokens is too low and the MRZ is being cut off; a low
        # layout rate means the layout prompt is not being honoured.
        "truncated_pages": sum(1 for item in diagnostics if item["finish_reason"] == "length"),
        "pages_with_layout": sum(1 for item in diagnostics if item["layout_blocks"]),
        "request_errors": sum(1 for item in diagnostics if item["error"]),
        "documents": len(diagnostics),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "near_misses": _tally(row["near_miss"] for row in supported if row["near_miss"]),
    }
    if args.type == "nid_back":
        overall["mrz_status"] = _tally(item["mrz_status"] or "absent" for item in diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "type": args.type, "split": args.split, "overall": overall,
        "per_field": per_field, "rows": rows, "diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.type} / {args.split} ===")
    print(f"{'field':<16}{'total':>7}{'correct':>9}{'accuracy':>10}")
    for field, stats in per_field.items():
        if not stats["total"]:
            continue
        print(f"{field:<16}{stats['total']:>7}{stats['correct']:>9}{stats['exact_match_rate'] * 100:>9.2f}%")
    print(f"{'OVERALL':<16}{overall['supported_field_total']:>7}{overall['supported_field_exact_matches']:>9}"
          f"{(overall['supported_field_exact_match_rate'] or 0) * 100:>9.2f}%")
    print(f"\nfalse positives on blank fields: {overall['blank_field_false_positives']}")
    print(f"truncated pages: {overall['truncated_pages']}   request errors: {overall['request_errors']}")
    if overall["near_misses"]:
        print(f"failure breakdown: {overall['near_misses']}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
