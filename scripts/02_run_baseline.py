import argparse
from pathlib import Path
from se_unlearning.utils.config import base_checkpoint_dir, load_config
from se_unlearning.utils.seeds import set_seed
from se_unlearning.utils.logging import setup_logger
from se_unlearning.data.load_dataset import load_jsonl, select_splits
from se_unlearning.models.load_model import load_model_and_tokenizer
from se_unlearning.models.checkpoints import (
    save_base_checkpoint,
    validate_base_checkpoint,
    validate_model_revision,
)
from se_unlearning.evaluation.experiment_runner import run_condition
from se_unlearning.utils.provenance import build_run_info

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    out = Path(cfg["experiment"]["output_dir"])
    rows = load_jsonl(cfg["data"]["input_jsonl"])
    limits = {
        "forget": cfg["data"].get("max_forget_examples"),
        "forget_update": cfg["data"].get("max_forget_update_examples"),
        "retain": cfg["data"].get("max_retain_examples"),
        "control": cfg["data"].get("max_control_examples"),
        "control_update": cfg["data"].get("max_control_update_examples"),
        "retain_update": cfg["data"].get("max_retain_update_examples"),
    }
    splits = select_splits(rows, limits)
    model_name = cfg["model"]["name_or_path"]
    revision = cfg["model"].get("revision", "main")
    validate_model_revision(model_name, revision)
    base_dir = base_checkpoint_dir(cfg)
    if cfg["experiment"].get("shared_base_dir") and not base_dir.exists():
        raise FileNotFoundError(f"Shared base checkpoint is missing: {base_dir}; run 01b first")
    if base_dir.exists():
        validate_base_checkpoint(base_dir, model_name, revision)
        model, tokenizer, device = load_model_and_tokenizer(
            str(base_dir), device=cfg["model"].get("device", "auto")
        )
    else:
        model, tokenizer, device = load_model_and_tokenizer(
            model_name, revision, cfg["model"].get("device", "auto")
        )
        save_base_checkpoint(model, tokenizer, base_dir, model_name, revision)
    manifest = validate_base_checkpoint(base_dir, model_name, revision)
    run_dir = out / "condition_no_update_original"
    logger = setup_logger(run_dir)
    logger.info("Loaded %s on %s", cfg["model"]["name_or_path"], device)
    run_condition(
        model,
        tokenizer,
        device,
        splits,
        [],
        cfg,
        run_dir,
        "no_update",
        "original",
        build_run_info(
            cfg,
            splits,
            device,
            manifest["resolved_revision"],
            manifest["checkpoint_sha256"],
        ),
        logger=logger,
    )
    logger.info("No-update baseline saved under %s", run_dir)

if __name__ == "__main__":
    main()
