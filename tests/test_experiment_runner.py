import json

from se_unlearning.evaluation.experiment_runner import run_condition
from se_unlearning.utils.artifacts import write_metric_rows


class TinyModel:
    def __init__(self, value=0):
        self.value = value

    def save_pretrained(self, path):
        (path / "value.json").write_text(json.dumps({"value": self.value}), encoding="utf-8")


class TinyTokenizer:
    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class TinyUnlearner:
    def __init__(self):
        self.calls = 0

    def update(self, model, tokenizer, rows, device, config, logger):
        self.calls += 1
        self.last_text = rows[0]["text"]
        model.value += 1
        return {"steps": 2}


def load_tiny_checkpoint(path):
    state = json.loads((__import__("pathlib").Path(path) / "value.json").read_text())
    return TinyModel(state["value"]), TinyTokenizer(), "cpu"


def fake_evaluate(model, tokenizer, splits, device, **kwargs):
    return {"cosine_separation": float(model.value)}


def runner_config():
    return {
        "experiment": {"seed": 12},
        "model": {"max_length": 8},
        "evaluation": {
            "fixed_layer": -1,
            "max_eval_examples_per_split": 2,
            "prompt_template": "Prompt: {text}",
        },
        "unlearning": {"max_steps_per_request": 2},
    }


def runner_info():
    return {
        "evaluation_fingerprint": "fixed-eval",
        "dataset_sha256": "fixed-dataset",
        "resolved_model_revision": "fixed-revision",
        "device": "cpu",
    }


def test_matched_update_count_and_resume_without_duplicate_updates(tmp_path):
    unlearner = TinyUnlearner()
    run_dir = tmp_path / "condition_forget_original"
    args = (
        "cpu",
        {"forget": [{"id": "f", "text": "f"}], "control": [], "retain": []},
        [{"id": "f1", "text": "f1"}, {"id": "f2", "text": "f2"}],
        runner_config(),
        run_dir,
        "forget",
        "original",
        runner_info(),
    )

    rows = run_condition(
        TinyModel(),
        TinyTokenizer(),
        *args[:1],
        *args[1:],
        unlearner=unlearner,
        resume_loader=load_tiny_checkpoint,
        evaluator=fake_evaluate,
    )
    assert [row["completed_update_count"] for row in rows] == [0, 1, 2]
    assert unlearner.calls == 2
    assert unlearner.last_text == "Prompt: f2"
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["update_count"] == 2
    assert metadata["optimizer_step_count"] == 4
    assert metadata["status"] == "complete"
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "metrics.json").is_file()

    resumed_rows = run_condition(
        TinyModel(),
        TinyTokenizer(),
        *args[:1],
        *args[1:],
        unlearner=unlearner,
        resume_loader=load_tiny_checkpoint,
        evaluator=fake_evaluate,
    )
    assert [row["step"] for row in resumed_rows] == [0, 1, 2]
    assert unlearner.calls == 2


def test_runner_recovers_metrics_after_checkpoint_save(tmp_path):
    unlearner = TinyUnlearner()
    run_dir = tmp_path / "condition_control_original"
    args = (
        "cpu",
        {"forget": [], "control": [{"id": "c", "text": "c"}], "retain": []},
        [{"id": "c1", "text": "c1"}],
        runner_config(),
        run_dir,
        "control",
        "original",
        runner_info(),
    )
    run_condition(
        TinyModel(),
        TinyTokenizer(),
        *args[:1],
        *args[1:],
        unlearner=unlearner,
        resume_loader=load_tiny_checkpoint,
        evaluator=fake_evaluate,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    write_metric_rows(run_dir / "metrics.csv", metrics[:-1])

    recovered = run_condition(
        TinyModel(),
        TinyTokenizer(),
        *args[:1],
        *args[1:],
        unlearner=unlearner,
        resume_loader=load_tiny_checkpoint,
        evaluator=fake_evaluate,
    )

    assert [row["step"] for row in recovered] == [0, 1]
    assert unlearner.calls == 1


def test_no_update_condition_performs_zero_updates(tmp_path):
    unlearner = TinyUnlearner()
    rows = run_condition(
        TinyModel(),
        TinyTokenizer(),
        "cpu",
        {"forget": [], "control": [], "retain": []},
        [],
        runner_config(),
        tmp_path / "condition_no_update_original",
        "no_update",
        "original",
        runner_info(),
        unlearner=unlearner,
        evaluator=fake_evaluate,
    )

    assert len(rows) == 1
    assert rows[0]["completed_update_count"] == 0
    assert unlearner.calls == 0
    metadata = json.loads(
        (tmp_path / "condition_no_update_original" / "run.json").read_text()
    )
    assert metadata["status"] == "complete"


def test_runner_records_update_failure_and_reraises(tmp_path):
    class FailingUnlearner:
        def update(self, *args):
            raise RuntimeError("synthetic update failure")

    import pytest

    run_dir = tmp_path / "condition_forget_original"
    with pytest.raises(RuntimeError, match="synthetic update failure"):
        run_condition(
            TinyModel(),
            TinyTokenizer(),
            "cpu",
            {"forget": [], "control": [], "retain": []},
            [{"id": "f", "text": "forget"}],
            runner_config(),
            run_dir,
            "forget",
            "original",
            runner_info(),
            unlearner=FailingUnlearner(),
            evaluator=fake_evaluate,
        )

    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "RuntimeError"


def resume_only_config():
    config = runner_config()
    config["experiment"]["checkpoint_retention"] = "resume_only"
    return config


def checkpoint_names(run_dir):
    return sorted(path.name for path in run_dir.glob("checkpoint_step_*"))


def test_resume_only_keeps_latest_checkpoint_and_removes_it_on_completion(tmp_path):
    class RecordingUnlearner(TinyUnlearner):
        def __init__(self, run_dir):
            super().__init__()
            self.run_dir = run_dir
            self.seen = []

        def update(self, model, tokenizer, rows, device, config, logger):
            self.seen.append(checkpoint_names(self.run_dir))
            return super().update(model, tokenizer, rows, device, config, logger)

    run_dir = tmp_path / "condition_forget_original"
    unlearner = RecordingUnlearner(run_dir)
    updates = [{"id": f"f{i}", "text": f"f{i}"} for i in range(1, 4)]
    args = (
        "cpu", {"forget": [], "control": [], "retain": []}, updates,
        resume_only_config(), run_dir, "forget", "original", runner_info(),
    )
    rows = run_condition(
        TinyModel(), TinyTokenizer(), *args,
        unlearner=unlearner, resume_loader=load_tiny_checkpoint, evaluator=fake_evaluate,
    )

    assert [row["cosine_separation"] for row in rows] == [0.0, 1.0, 2.0, 3.0]
    assert unlearner.seen == [[], ["checkpoint_step_1"], ["checkpoint_step_2"]]
    assert checkpoint_names(run_dir) == []
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"

    rerun = run_condition(
        TinyModel(), TinyTokenizer(), *args,
        unlearner=unlearner, resume_loader=load_tiny_checkpoint, evaluator=fake_evaluate,
    )
    assert [row["step"] for row in rerun] == [0, 1, 2, 3]
    assert unlearner.calls == 3


def test_resume_only_resumes_from_latest_after_failure(tmp_path):
    class FailOnceAtStepThree(TinyUnlearner):
        def update(self, model, tokenizer, rows, device, config, logger):
            if rows[0]["id"] == "f3" and not getattr(self, "failed", False):
                self.failed = True
                raise RuntimeError("interrupted")
            return super().update(model, tokenizer, rows, device, config, logger)

    import pytest

    run_dir = tmp_path / "condition_forget_original"
    unlearner = FailOnceAtStepThree()
    updates = [{"id": f"f{i}", "text": f"f{i}"} for i in range(1, 4)]
    args = (
        "cpu", {"forget": [], "control": [], "retain": []}, updates,
        resume_only_config(), run_dir, "forget", "original", runner_info(),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run_condition(
            TinyModel(), TinyTokenizer(), *args,
            unlearner=unlearner, resume_loader=load_tiny_checkpoint, evaluator=fake_evaluate,
        )
    assert checkpoint_names(run_dir) == ["checkpoint_step_2"]

    rows = run_condition(
        TinyModel(), TinyTokenizer(), *args,
        unlearner=unlearner, resume_loader=load_tiny_checkpoint, evaluator=fake_evaluate,
    )
    assert [row["cosine_separation"] for row in rows] == [0.0, 1.0, 2.0, 3.0]
    assert unlearner.calls == 3
    assert checkpoint_names(run_dir) == []


def test_unknown_checkpoint_retention_is_rejected(tmp_path):
    import pytest

    config = runner_config()
    config["experiment"]["checkpoint_retention"] = "some"
    with pytest.raises(ValueError, match="checkpoint_retention"):
        run_condition(
            TinyModel(), TinyTokenizer(), "cpu", {"forget": [], "control": [], "retain": []},
            [], config, tmp_path / "run", "no_update", "original", runner_info(),
            evaluator=fake_evaluate,
        )


def test_per_example_losses_are_stored_per_step_not_in_metric_rows(tmp_path):
    def evaluate_with_examples(model, tokenizer, splits, device, **kwargs):
        return {
            "cosine_separation": float(model.value),
            "per_example": {"forget": {"f": float(model.value)}},
        }

    run_dir = tmp_path / "condition_forget_original"
    rows = run_condition(
        TinyModel(), TinyTokenizer(), "cpu", {"forget": [], "control": [], "retain": []},
        [{"id": "f1", "text": "f1"}, {"id": "f2", "text": "f2"}], runner_config(), run_dir,
        "forget", "original", runner_info(), unlearner=TinyUnlearner(),
        resume_loader=load_tiny_checkpoint, evaluator=evaluate_with_examples,
    )
    assert all("per_example" not in row for row in rows)
    for step in (0, 1, 2):
        stored = json.loads((run_dir / "per_example" / f"step_{step}.json").read_text())
        assert stored == {"forget": {"f": float(step)}}
