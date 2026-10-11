import pytest
from se_unlearning.evaluation.controls import (
    make_request_sequence,
    match_update_rows_by_tokens,
    match_updates_to_reference_order,
    matched_order,
    shuffled_copy,
    select_update_rows,
)

def test_sequence():
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert len(make_request_sequence(rows, 2)) == 2

def test_bad_order_fails():
    with pytest.raises(ValueError):
        matched_order([{"id":"a"}, {"id":"b"}], [0, 0])

def test_shuffled_copy_preserves_source_rows():
    rows = [{"id": str(i)} for i in range(8)]

    shuffled = shuffled_copy(rows, 18)

    assert rows == [{"id": str(i)} for i in range(8)]
    assert sorted(row["id"] for row in shuffled) == [str(i) for i in range(8)]
    assert shuffled == shuffled_copy(rows, 18)

def test_update_conditions_select_equal_deterministic_counts():
    splits = {
        "forget_update": [{"id": f"forget-{i}"} for i in range(4)],
        "control_update": [{"id": f"control-{i}"} for i in range(4)],
        "retain_update": [{"id": f"retain-{i}"} for i in range(4)],
    }

    forget = select_update_rows(splits, "forget", 3, "original", 8)
    control = select_update_rows(splits, "control", 3, "original", 8)
    shuffled = select_update_rows(splits, "forget", 3, "shuffled", 8)

    assert len(forget) == len(control) == len(shuffled) == 3
    assert [row["id"] for row in forget] == ["forget-0", "forget-1", "forget-2"]
    assert [row["id"] for row in control] == ["control-0", "control-1", "control-2"]
    assert sorted(row["id"] for row in shuffled) == [row["id"] for row in forget]

def test_update_condition_requires_matched_rows():
    splits = {
        "forget_update": [{"id": "f"}],
        "control_update": [],
        "retain_update": [],
    }

    with pytest.raises(ValueError, match="control_update"):
        select_update_rows(splits, "control", 1, "original", 0)

def test_control_rows_are_paired_by_prompt_token_length_without_reordering_text():
    class WhitespaceTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": text.split()}

    forget = [
        {"id": "f-short", "text": "one two"},
        {"id": "f-long", "text": "one two three four"},
    ]
    controls = [
        {"id": "c-long", "text": "a b c d"},
        {"id": "c-short", "text": "a b"},
        {"id": "c-mid", "text": "a b c"},
    ]

    matched, forget_lengths, control_lengths = match_update_rows_by_tokens(
        forget, controls, WhitespaceTokenizer(), 20, "Q: {text}"
    )

    assert [row["id"] for row in matched] == ["c-short", "c-long"]
    assert forget_lengths == control_lengths == [3, 5]
    assert matched[0]["text"] == "a b"


def test_ordered_controls_keep_same_request_pairs_across_order_conditions():
    class WhitespaceTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": text.split()}

    forget = [
        {"id": "f-short", "text": "one two"},
        {"id": "f-long", "text": "one two three four"},
    ]
    candidates = [
        {"id": "c-long", "text": "a b c d"},
        {"id": "c-short", "text": "a b"},
        {"id": "c-mid", "text": "a b c"},
    ]
    original = match_updates_to_reference_order(
        forget, candidates, forget, WhitespaceTokenizer(), 20, "Q: {text}"
    )
    shuffled = match_updates_to_reference_order(
        forget,
        candidates,
        list(reversed(forget)),
        WhitespaceTokenizer(),
        20,
        "Q: {text}",
    )

    assert [row["id"] for row in original[0]] == ["c-short", "c-long"]
    assert [row["id"] for row in shuffled[0]] == ["c-long", "c-short"]
    assert shuffled[1:] == ([5, 3], [5, 3])
