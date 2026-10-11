import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_metric_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Metrics JSON must contain a list of objects: {path}")
    return rows


def write_metric_rows(csv_path: str | Path, rows: list[dict]) -> None:
    csv_path = Path(csv_path)
    json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")
    frame.to_csv(csv_tmp, index=False)
    json_tmp.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    csv_tmp.replace(csv_path)
    json_tmp.replace(json_path)
