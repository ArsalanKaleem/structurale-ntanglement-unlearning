import importlib.util
import json
from pathlib import Path

import pytest


def load_analyze():
    script = Path(__file__).parents[1] / "scripts" / "06_analyze.py"
    spec = importlib.util.spec_from_file_location("pilot_analysis", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


def write_condition(root, condition, seed, step_value):
    run_dir = root / f"condition_{condition}_original_seed{seed}"
    run_dir.mkdir(parents=True)
    metadata = {
        "condition": condition,
        "order": "original",
        "seed": seed,
        "dataset_sha256": "dataset-hash",
        "resolved_model_revision": "model-commit",
        "base_checkpoint_sha256": "base-hash",
        "evaluation_fingerprint": "evaluation-hash",
        "hyperparameters": {"learning_rate": 0.01},
        "evaluation_settings": {"prompt_template": "{text}"},
        "status": "complete",
        "update_count": 0 if step_value is None else 1,
        "optimizer_step_count": 0 if step_value is None else 2,
        "reference_forget_token_counts": None if step_value is None else [4],
    }
    (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    baseline = {
        "step": 0,
        "cosine_separation": 0.25,
    }
    metrics = [baseline]
    if step_value is not None:
        metrics.append({"step": 1, "cosine_separation": step_value})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_analysis_reports_paired_difference_in_change_and_writes_csv_json(tmp_path):
    root = tmp_path / "runs"
    for seed, effect in ((3, 0.1), (9, 0.3)):
        write_condition(root, "no_update", seed, None)
        write_condition(root, "forget", seed, 0.25 + effect + 0.02)
        write_condition(root, "control", seed, 0.27)
        write_condition(root, "retain", seed, 0.25 + effect)

    summaries = load_analyze()(root, tmp_path / "analysis", bootstrap_samples=500)

    control_effect = next(
        summary
        for summary in summaries
        if summary["control_condition"] == "control"
    )
    retain_effect = next(
        summary
        for summary in summaries
        if summary["control_condition"] == "retain"
    )
    no_update_effect = next(
        summary
        for summary in summaries
        if summary["control_condition"] == "no_update"
    )
    assert control_effect["mean_paired_effect"] == pytest.approx(0.2)
    assert retain_effect["mean_paired_effect"] == pytest.approx(0.02)
    assert no_update_effect["mean_paired_effect"] == pytest.approx(0.22)
    assert control_effect["n_seeds"] == 2
    assert control_effect["bootstrap_95_percentile_interval"] is not None
    assert (tmp_path / "analysis" / "analysis.csv").is_file()
    report = json.loads((tmp_path / "analysis" / "analysis.json").read_text())
    assert report["primary_metric"] == "cosine_separation"


def test_analysis_rejects_mismatched_base_checkpoint(tmp_path):
    root = tmp_path / "runs"
    write_condition(root, "no_update", 2, None)
    write_condition(root, "forget", 2, 0.4)
    write_condition(root, "control", 2, 0.3)
    write_condition(root, "retain", 2, 0.3)
    control_metrics = root / "condition_control_original_seed2" / "run.json"
    metadata = json.loads(control_metrics.read_text())
    metadata["base_checkpoint_sha256"] = "different"
    control_metrics.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        load_analyze()(root, tmp_path / "analysis", bootstrap_samples=10)


def test_analysis_rejects_mismatched_forget_token_sequence(tmp_path):
    root = tmp_path / "runs"
    write_condition(root, "no_update", 2, None)
    write_condition(root, "forget", 2, 0.4)
    write_condition(root, "control", 2, 0.3)
    write_condition(root, "retain", 2, 0.3)
    control_metadata = root / "condition_control_original_seed2" / "run.json"
    metadata = json.loads(control_metadata.read_text())
    metadata["reference_forget_token_counts"] = [2]
    control_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Forget reference token lengths"):
        load_analyze()(root, tmp_path / "analysis", bootstrap_samples=10)


def test_secondary_metric_analysis_writes_separate_files(tmp_path):
    root = tmp_path / "runs"
    for seed in (3, 9):
        write_condition(root, "no_update", seed, None)
        write_condition(root, "forget", seed, 0.4)
        write_condition(root, "control", seed, 0.3)
        write_condition(root, "retain", seed, 0.3)
    for metrics_path in root.rglob("metrics.json"):
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in rows:
            row["cosine_separation_forget_retain"] = row["cosine_separation"] * 2
        metrics_path.write_text(json.dumps(rows), encoding="utf-8")

    summaries = load_analyze()(
        root, tmp_path / "analysis", bootstrap_samples=50,
        metric="cosine_separation_forget_retain",
    )

    control_effect = next(s for s in summaries if s["control_condition"] == "control")
    assert control_effect["mean_paired_effect"] == pytest.approx(0.2)
    report = json.loads(
        (tmp_path / "analysis" / "analysis_cosine_separation_forget_retain.json").read_text()
    )
    assert report["metric"] == "cosine_separation_forget_retain"
    assert report["primary_metric"] == "cosine_separation"
    assert not (tmp_path / "analysis" / "analysis.json").exists()
