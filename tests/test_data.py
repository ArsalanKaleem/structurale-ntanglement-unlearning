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

def test_split_selection_is_deterministic_and_preserves_file_order():
    rows = [
        {"id": "f2", "text": "f2", "split": "forget"},
        {"id": "f1", "text": "f1", "split": "forget"},
        {"id": "r1", "text": "r1", "split": "retain"},
        {"id": "c1", "text": "c1", "split": "control"},
    ]

    first = select_splits(rows, {"forget": 1})
    second = select_splits(rows, {"forget": 1})

    assert first == second
    assert list(first) == [
        "forget",
        "retain",
        "control",
        "forget_update",
        "control_update",
        "retain_update",
    ]
    assert [row["id"] for row in first["forget"]] == ["f2"]

def test_duplicate_normalized_text_across_splits_is_rejected():
    rows = [
        {"id": "f", "text": "Same content", "split": "forget"},
        {"id": "r", "text": "retain", "split": "retain"},
        {"id": "c", "text": "control", "split": "control"},
        {"id": "u", "text": " same CONTENT ", "split": "control_update"},
    ]

    with pytest.raises(ValueError, match="Exact normalized text overlap"):
        select_splits(rows)
