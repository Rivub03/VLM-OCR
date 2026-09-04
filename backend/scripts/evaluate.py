"""Evaluate OCR JSON outputs against redacted ground-truth JSON fixtures.

Expected input: a directory of pairs named <sample>.expected.json and
<sample>.actual.json. Each file contains `{ "text": "...", "fields": {...} }`.
"""
import argparse
import json
from pathlib import Path


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, char in enumerate(left, start=1):
        current = [row]
        for column, other in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (char != other)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    expected_files = sorted(args.fixtures.glob("*.expected.json"))
    if not expected_files:
        raise SystemExit("No .expected.json fixtures were found.")
    char_errors = char_total = fields_correct = fields_total = 0
    for expected_path in expected_files:
        actual_path = expected_path.with_name(expected_path.name.replace(".expected.json", ".actual.json"))
        if not actual_path.exists():
            print(f"SKIP {expected_path.name}: no actual output")
            continue
        expected, actual = json.loads(expected_path.read_text(encoding="utf-8")), json.loads(actual_path.read_text(encoding="utf-8"))
        truth, observed = expected.get("text", ""), actual.get("text", "")
        char_errors += distance(truth, observed)
        char_total += max(1, len(truth))
        for key, value in expected.get("fields", {}).items():
            fields_total += 1
            fields_correct += actual.get("fields", {}).get(key) == value
    print(json.dumps({
        "character_error_rate": round(char_errors / char_total, 6),
        "field_exact_match": round(fields_correct / fields_total, 6) if fields_total else None,
        "field_count": fields_total,
    }, indent=2))


if __name__ == "__main__":
    main()

