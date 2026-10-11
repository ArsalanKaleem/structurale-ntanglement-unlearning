import pytest
from se_unlearning.evaluation.controls import (
    make_request_sequence,
    matched_order,
    shuffled_copy,
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
