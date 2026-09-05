"""Evaluate the service against an external, untracked NID benchmark folder."""

import argparse
import json
import mimetypes
import os
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


def normalise(value: Any, field: str) -> str:
    value = " ".join(str(value or "").strip().split())
    if field == "nid_no":
        return "".join(character for character in value if character.isdigit())
    return value.casefold() if field in {"name", "place_of_birth"} else value


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--type", choices=FIELD_MAP, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.getenv("OCR_API_KEY"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Provide --api-key or OCR_API_KEY.")

    documents = documents_from_ground_truth(args.benchmark_root / "data" / "ground_truth" / f"{args.type}_{args.split}.json")
    image_dir = args.benchmark_root / "data" / "images" / args.type / args.split
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=300) as client:
        for image_id, document in documents.items():
            image = image_for(image_dir, image_id)
            with image.open("rb") as source:
                response = client.post(
                    f"{args.base_url.rstrip('/')}/api/v1/ocr",
                    headers={"X-API-Key": args.api_key},
                    files={"file": (image.name, source, mimetypes.guess_type(image.name)[0] or "application/octet-stream")},
                    data={"mode": args.type},
                )
            response.raise_for_status()
            fields = response.json()["result"][0]["fields"] or {}
            truth = document.get("fields", {})
            for predicted_key, truth_key in FIELD_MAP[args.type].items():
                expected = truth.get(truth_key)
                actual = fields.get(predicted_key)
                rows.append({
                    "image_id": image_id,
                    "field": predicted_key,
                    "expected": expected,
                    "actual": actual,
                    "matched": normalise(actual, predicted_key) == normalise(expected, predicted_key) and bool(normalise(expected, predicted_key)),
                    "false_positive": not normalise(expected, predicted_key) and bool(normalise(actual, predicted_key)),
                    "null": actual is None,
                })
    per_field = {}
    for field in FIELD_MAP[args.type]:
        values = [row for row in rows if row["field"] == field and normalise(row["expected"], field)]
        per_field[field] = {
            "total": len(values),
            "exact_match_rate": sum(row["matched"] for row in values) / len(values) if values else None,
            "null_rate": sum(row["null"] for row in values) / len(values) if values else None,
            "false_positives": sum(row["false_positive"] for row in rows if row["field"] == field),
        }
    supported = [row for row in rows if normalise(row["expected"], row["field"])]
    blank = [row for row in rows if not normalise(row["expected"], row["field"])]
    overall = {
        "supported_field_total": len(supported),
        "supported_field_exact_matches": sum(row["matched"] for row in supported),
        "supported_field_exact_match_rate": sum(row["matched"] for row in supported) / len(supported) if supported else None,
        "blank_field_total": len(blank),
        "blank_field_false_positives": sum(row["false_positive"] for row in blank),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"type": args.type, "split": args.split, "overall": overall, "per_field": per_field, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
