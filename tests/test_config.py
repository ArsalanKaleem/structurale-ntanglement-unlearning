import pytest

from se_unlearning.utils.config import load_config


def config_text(metric="cosine_separation"):
    return f"""
experiment: {{name: test, seed: 1, output_dir: results}}
model: {{name_or_path: model, revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}}
data: {{input_jsonl: data.jsonl, version: test-v1}}
unlearning: {{method: ga}}
evaluation:
  primary_structure_metric: {metric}
  prompt_template: "{{text}}"
"""


def test_config_freezes_primary_metric_and_requires_prompt_field(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(config_text(), encoding="utf-8")

    assert load_config(path)["evaluation"]["primary_structure_metric"] == "cosine_separation"

    path.write_text(config_text("another_metric"), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen primary_structure_metric"):
        load_config(path)


def test_config_rejects_placeholder_dataset_version_before_model_loading(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(config_text().replace("test-v1", "REPLACE_WITH_DATASET_VERSION"), encoding="utf-8")

    with pytest.raises(ValueError, match="data.version"):
        load_config(path)
