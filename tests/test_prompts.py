import pytest

from se_unlearning.data.prompts import render_rows


def test_prompt_rendering_is_deterministic_and_preserves_ids():
    rows = [
        {"id": "forget-1", "text": "Remember this.", "split": "forget"},
        {"id": "control-1", "text": "Unrelated text.", "split": "control"},
    ]

    result = render_rows(rows, "Question: {text}\nAnswer:")

    assert [row["id"] for row in result] == ["forget-1", "control-1"]
    assert result[0]["text"] == "Question: Remember this.\nAnswer:"
    assert rows[0]["text"] == "Remember this."


@pytest.mark.parametrize("template", ["plain text", "{missing}", "{text!unknown}"])
def test_invalid_prompt_template_fails(template):
    with pytest.raises(ValueError, match="prompt_template"):
        render_rows([{"id": "x", "text": "text"}], template)
