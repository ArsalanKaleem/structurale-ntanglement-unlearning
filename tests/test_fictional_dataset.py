import importlib.util
import json
from collections import Counter
from pathlib import Path

from se_unlearning.data.load_dataset import load_jsonl, select_splits

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_fictional_dataset", ROOT / "scripts" / "00_generate_fictional_dataset.py"
)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_generator_is_deterministic_and_matches_committed_file():
    rows = generator.build_rows()
    assert rows == generator.build_rows()
    committed = load_jsonl(ROOT / "data" / "fictional_authors_v1.jsonl")
    assert committed == json.loads(json.dumps(rows))


def test_generated_splits_are_balanced_and_disjoint(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in generator.build_rows()), encoding="utf-8")
    splits = select_splits(load_jsonl(path))
    assert {name: len(rows) for name, rows in splits.items()} == {
        "forget": 64, "retain": 64, "control": 64,
        "forget_update": 64, "control_update": 64, "retain_update": 64,
    }
    forget_entities = {r["entity"] for r in splits["forget"]}
    assert forget_entities == {r["entity"] for r in splits["forget_update"]}
    assert forget_entities.isdisjoint(r["entity"] for r in splits["retain"])
    assert forget_entities.isdisjoint(r["entity"] for r in splits["control"])
    first_requests = [r["entity"] for r in splits["forget_update"][:16]]
    assert len(set(first_requests)) == 16
    assert set(Counter(r["entity"] for r in splits["forget"]).values()) == {4}
