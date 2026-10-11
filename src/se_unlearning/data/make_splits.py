import json
from pathlib import Path
from .load_dataset import load_jsonl, select_splits

def prepare(input_path: str, output_dir: str, limits: dict | None = None) -> dict:
    rows = load_jsonl(input_path)
    splits = select_splits(rows, limits)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {k: len(v) for k, v in splits.items()}
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
