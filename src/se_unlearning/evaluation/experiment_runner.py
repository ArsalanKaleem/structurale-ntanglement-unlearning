import json
import logging
import shutil
from pathlib import Path

from se_unlearning.evaluation.evaluate_checkpoint import evaluate
from se_unlearning.data.prompts import render_rows
from se_unlearning.utils.artifacts import (
    load_metric_rows,
    stable_fingerprint,
    write_metric_rows,
    write_json_atomic,
)
from se_unlearning.utils.logging import write_run_metadata
from se_unlearning.utils.seeds import set_seed


CHECKPOINT_RETENTION_MODES = ("all", "resume_only")


def checkpoint_retention(config: dict) -> str:
    mode = config.get("experiment", {}).get("checkpoint_retention", "all")
    if mode not in CHECKPOINT_RETENTION_MODES:
        raise ValueError(
            f"experiment.checkpoint_retention must be one of {CHECKPOINT_RETENTION_MODES}; "
            f"got {mode!r}"
        )
    return mode


def _checkpoint_steps(run_dir: Path) -> list[int]:
    steps = []
    for path in run_dir.glob("checkpoint_step_*"):
        suffix = path.name.removeprefix("checkpoint_step_")
        if suffix.isdigit() and path.is_dir():
            steps.append(int(suffix))
    return sorted(steps)


def latest_checkpoint_step(run_dir: str | Path, pruned: bool = False) -> int:
    """Return the latest saved update step.

    With ``pruned=True`` (``resume_only`` retention) earlier checkpoints may have been
    deleted, so only the surviving checkpoints must be contiguous.
    """
    run_dir = Path(run_dir)
    steps = _checkpoint_steps(run_dir)
    first = steps[0] if pruned and steps else 1
    if steps and steps != list(range(first, max(steps) + 1)):
        raise ValueError(f"Checkpoint sequence has gaps under {run_dir}")
    return max(steps, default=0)


def prune_checkpoints(run_dir: str | Path, keep_step: int | None) -> None:
    """Delete step checkpoints other than ``keep_step`` (all of them when None)."""
    run_dir = Path(run_dir)
    for step in _checkpoint_steps(run_dir):
        if keep_step is None or step < keep_step:
            shutil.rmtree(run_dir / f"checkpoint_step_{step}")


def run_condition(
    model,
    tokenizer,
    device: str,
    splits: dict[str, list[dict]],
    update_rows: list[dict],
    config: dict,
    run_dir: str | Path,
    condition: str,
    order: str,
    run_info: dict,
    unlearner=None,
    resume_loader=None,
    restored_checkpoint_step: int | None = None,
    evaluator=evaluate,
    logger: logging.Logger | None = None,
) -> list[dict]:
    if update_rows and unlearner is None:
        raise ValueError("An unlearner is required when update rows are provided")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    retention = checkpoint_retention(config)
    metrics_path = run_dir / "metrics.csv"
    metadata_path = run_dir / "run.json"
    fingerprint = stable_fingerprint(
        {
            "config": config,
            "condition": condition,
            "order": order,
            "evaluation": run_info.get("evaluation_fingerprint"),
            "dataset_sha256": run_info.get("dataset_sha256"),
            "base_model_revision": run_info.get("resolved_model_revision"),
            "base_checkpoint_sha256": run_info.get("base_checkpoint_sha256"),
            "update_sequence_fingerprint": run_info.get("update_sequence_fingerprint"),
        }
    )
    if metadata_path.exists():
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_metadata.get("experiment_fingerprint") != fingerprint:
            raise ValueError(
                f"Run configuration differs from existing artifacts in {run_dir}; "
                "use a new output directory."
            )
        old_steps = [int(row["step"]) for row in load_metric_rows(run_dir / "metrics.json")]
        if (
            old_metadata.get("status") == "complete"
            and old_metadata.get("update_count") == len(update_rows)
            and old_steps == list(range(len(update_rows) + 1))
        ):
            if retention == "resume_only":
                prune_checkpoints(run_dir, keep_step=None)
            return load_metric_rows(run_dir / "metrics.json")
    else:
        write_run_metadata(
            run_dir,
            config,
            {
                **run_info,
                "condition": condition,
                "order": order,
                "experiment_fingerprint": fingerprint,
                "update_count": 0,
                "optimizer_step_count": 0,
                "status": "in_progress",
            },
        )

    completed = latest_checkpoint_step(run_dir, pruned=retention == "resume_only")
    if completed > len(update_rows):
        raise ValueError("Saved checkpoint count exceeds the configured update sequence")
    if completed:
        if restored_checkpoint_step != completed:
            if resume_loader is None:
                raise ValueError("A resume_loader is required to restore saved checkpoints")
            model, tokenizer, device = resume_loader(
                str(run_dir / f"checkpoint_step_{completed}")
            )

    rows = load_metric_rows(run_dir / "metrics.json")
    metric_steps = [int(row["step"]) for row in rows]
    if metric_steps and metric_steps != list(range(max(metric_steps) + 1)):
        raise ValueError(f"Metric steps are not contiguous in {run_dir}")
    if metric_steps and max(metric_steps) > completed:
        raise ValueError("Metrics claim updates beyond the last saved checkpoint")
    if completed and not rows:
        raise ValueError("Saved checkpoints exist but the baseline metrics are missing")
    if completed and max(metric_steps) < completed - 1:
        raise ValueError("More than one checkpoint metric is missing; refusing unsafe resume")

    eval_kwargs = {
        "max_length": int(config["model"].get("max_length", 256)),
        "fixed_layer": int(config["evaluation"].get("fixed_layer", -1)),
        "max_examples_per_split": int(
            config["evaluation"].get("max_eval_examples_per_split", 64)
        ),
        "prompt_template": config["evaluation"].get("prompt_template", "{text}"),
    }
    update_rows = render_rows(update_rows, eval_kwargs["prompt_template"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "in_progress"
    metadata.pop("error_type", None)
    metadata.pop("error_message", None)
    write_json_atomic(metadata_path, metadata)
    if not rows:
        try:
            baseline = evaluator(model, tokenizer, splits, device, **eval_kwargs)
        except Exception as exc:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "failed"
            metadata["error_type"] = type(exc).__name__
            metadata["error_message"] = str(exc)
            write_json_atomic(metadata_path, metadata)
            raise
        rows = [
            {
                "condition": condition,
                "order": order,
                "step": 0,
                "completed_update_count": 0,
                "request_id": "baseline",
                "update_result": "",
                **baseline,
            }
        ]
        write_metric_rows(metrics_path, rows)

    if completed and max(metric_steps) < completed:
        state_path = run_dir / f"checkpoint_step_{completed}" / "update.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"Checkpoint update state is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        try:
            metrics = evaluator(model, tokenizer, splits, device, **eval_kwargs)
        except Exception as exc:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "failed"
            metadata["error_type"] = type(exc).__name__
            metadata["error_message"] = str(exc)
            write_json_atomic(metadata_path, metadata)
            raise
        rows.append(
            {
                "condition": condition,
                "order": order,
                "step": completed,
                "completed_update_count": completed,
                "request_id": state["request_id"],
                "update_result": json.dumps(state["update_result"], sort_keys=True),
                **metrics,
            }
        )
        write_metric_rows(metrics_path, rows)
        if retention == "resume_only":
            prune_checkpoints(run_dir, keep_step=completed)

    try:
        for step in range(completed + 1, len(update_rows) + 1):
            row = update_rows[step - 1]
            update_config = dict(config["unlearning"])
            update_config["max_length"] = eval_kwargs["max_length"]
            set_seed(int(config["experiment"]["seed"]) + step)
            update_result = unlearner.update(
                model, tokenizer, [row], device, update_config, logger
            )
            final_checkpoint = run_dir / f"checkpoint_step_{step}"
            temporary_checkpoint = run_dir / f".checkpoint_step_{step}.tmp"
            if temporary_checkpoint.exists():
                shutil.rmtree(temporary_checkpoint)
            temporary_checkpoint.mkdir()
            model.save_pretrained(temporary_checkpoint)
            tokenizer.save_pretrained(temporary_checkpoint)
            request_id = row.get("request_id", row["id"])
            (temporary_checkpoint / "update.json").write_text(
                json.dumps(
                    {"request_id": request_id, "update_result": update_result}, indent=2
                ),
                encoding="utf-8",
            )
            temporary_checkpoint.replace(final_checkpoint)
            metrics = evaluator(model, tokenizer, splits, device, **eval_kwargs)
            rows.append(
                {
                    "condition": condition,
                    "order": order,
                    "step": step,
                    "completed_update_count": step,
                    "request_id": request_id,
                    "update_result": json.dumps(update_result, sort_keys=True),
                    **metrics,
                }
            )
            write_metric_rows(metrics_path, rows)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["update_count"] = step
            metadata["optimizer_step_count"] = sum(
                int(json.loads(item["update_result"]).get("steps", 0))
                for item in rows
                if item["update_result"]
            )
            write_json_atomic(metadata_path, metadata)
            if retention == "resume_only":
                prune_checkpoints(run_dir, keep_step=step)
            if logger:
                logger.info("Condition %s step %d metrics: %s", condition, step, metrics)
    except Exception as exc:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        metadata["error_message"] = str(exc)
        write_json_atomic(metadata_path, metadata)
        raise
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["update_count"] = max((int(row["step"]) for row in rows), default=0)
    metadata["optimizer_step_count"] = sum(
        int(json.loads(row["update_result"]).get("steps", 0))
        for row in rows
        if row["update_result"]
    )
    metadata["status"] = "complete"
    write_json_atomic(metadata_path, metadata)
    if retention == "resume_only":
        prune_checkpoints(run_dir, keep_step=None)
    return rows
