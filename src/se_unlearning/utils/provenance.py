import hashlib
from pathlib import Path

from se_unlearning.data.prompts import render_rows
from se_unlearning.utils.artifacts import sha256_file, stable_fingerprint


def build_run_info(
    config: dict,
    splits: dict[str, list[dict]],
    device: str,
    resolved_revision: str,
    base_checkpoint_sha256: str,
):
    data_path = Path(config["data"]["input_jsonl"])
    data_version = config["data"].get("version")
    if not data_version or str(data_version).lower().startswith("replace"):
        raise ValueError("Set data.version to a fixed dataset version before running")

    evaluation = config["evaluation"]
    prompt_template = evaluation.get("prompt_template", "{text}")
    max_examples = int(evaluation.get("max_eval_examples_per_split", 64))
    evaluated_rows = {
        name: render_rows(splits[name][:max_examples], prompt_template)
        for name in ("forget", "retain", "control")
    }
    evaluation_manifest = {
        name: [
            {
                "id": row["id"],
                "prompt_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            }
            for row in rows
        ]
        for name, rows in evaluated_rows.items()
    }
    return {
        "device": device,
        "requested_model_revision": config["model"].get("revision", "main"),
        "resolved_model_revision": resolved_revision,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "dataset_version": data_version,
        "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "seed": int(config["experiment"]["seed"]),
        "update_seed_scheme": "experiment.seed + one-based request update index",
        "hyperparameters": config["unlearning"],
        "evaluation_settings": {
            "primary_metric": evaluation.get("primary_structure_metric"),
            "prompt_template": prompt_template,
            "fixed_layer": int(evaluation.get("fixed_layer", -1)),
            "max_examples_per_split": max_examples,
            "max_length": int(config["model"].get("max_length", 256)),
        },
        "evaluation_examples": evaluation_manifest,
        "evaluation_fingerprint": stable_fingerprint(evaluation_manifest),
    }
