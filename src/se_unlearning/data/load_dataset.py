import json
from pathlib import Path

VALID_SPLITS = {"forget", "retain", "control"}

def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Line {line_no} must contain a JSON object")
            for key in ("id", "text", "split"):
                if key not in row:
                    raise ValueError(f"Line {line_no} missing required key {key!r}")
            if not isinstance(row["id"], str) or not row["id"].strip():
                raise ValueError(f"Line {line_no}: id must be a non-empty string")
            if row["split"] not in VALID_SPLITS:
                raise ValueError(f"Line {line_no}: split must be one of {sorted(VALID_SPLITS)}")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"Line {line_no}: text must be a non-empty string")
            rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset IDs must be unique.")
    if not rows:
        raise ValueError("Dataset is empty.")
    return rows

def select_splits(rows: list[dict], limits: dict | None = None) -> dict[str, list[dict]]:
    limits = limits or {}
    result = {}
    for split in VALID_SPLITS:
        subset = [r for r in rows if r["split"] == split]
        limit = limits.get(split)
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise ValueError(f"Limit for split {split!r} must be >= 0")
            subset = subset[:limit]
        result[split] = subset
    if not result["forget"]:
        raise ValueError("At least one row with split='forget' is required.")
    if not result["retain"]:
        raise ValueError("At least one row with split='retain' is required.")
    if not result["control"]:
        raise ValueError("At least one row with split='control' is required.")
    return result
