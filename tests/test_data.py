import json
import pytest
from se_unlearning.data.load_dataset import load_jsonl, select_splits

def test_load_and_split(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"id":"f1","text":"forget text","split":"forget"},
        {"id":"r1","text":"retain text","split":"retain"},
        {"id":"c1","text":"control text","split":"control"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    loaded = load_jsonl(p)
    splits = select_splits(loaded)
    assert len(splits["forget"]) == 1
    assert len(splits["retain"]) == 1
    assert len(splits["control"]) == 1

def test_duplicate_ids_fail(tmp_path):
    p = tmp_path / "x.jsonl"
    row = {"id":"x","text":"text","split":"forget"}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_jsonl(p)

def test_zero_split_limit_does_not_mean_unlimited(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"id": "f", "text": "forget text", "split": "forget"},
        {"id": "r", "text": "retain text", "split": "retain"},
        {"id": "c", "text": "control text", "split": "control"},
    ]
    p.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="split='forget'"):
        select_splits(load_jsonl(p), {"forget": 0})

def test_non_object_jsonl_row_fails(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("null\n", encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        load_jsonl(p)
