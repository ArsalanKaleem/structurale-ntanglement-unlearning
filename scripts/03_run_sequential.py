import argparse
from pathlib import Path
from se_unlearning.utils.config import load_config
from se_unlearning.utils.seeds import set_seed
from se_unlearning.utils.logging import setup_logger
from se_unlearning.data.load_dataset import load_jsonl, select_splits
from se_unlearning.models.load_model import load_model_and_tokenizer
from se_unlearning.models.checkpoints import validate_base_checkpoint
from se_unlearning.unlearning.gradient_ascent import build_unlearner
from se_unlearning.evaluation.controls import (
    UPDATE_SPLIT_BY_CONDITION,
    count_token_lengths,
    match_updates_to_reference_order,
    select_update_rows,
)
from se_unlearning.evaluation.experiment_runner import (
    checkpoint_retention,
    latest_checkpoint_step,
    run_condition,
)
from se_unlearning.utils.provenance import build_run_info
from se_unlearning.utils.artifacts import stable_fingerprint

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--condition", choices=["forget", "control", "retain"], default="forget")
    p.add_argument("--order", choices=["original", "shuffled"], default="original")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
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
    seq_len = int(
        cfg["unlearning"].get(
            "sequence_length", len(splits[UPDATE_SPLIT_BY_CONDITION["forget"]])
        )
    )
    base_dir = out / "base_model"
    manifest = validate_base_checkpoint(
        base_dir,
        cfg["model"]["name_or_path"],
        cfg["model"].get("revision", "main"),
    )
    run_dir = out / f"condition_{args.condition}_{args.order}"
    resume_step = latest_checkpoint_step(
        run_dir, pruned=checkpoint_retention(cfg) == "resume_only"
    )
    checkpoint_source = (
        run_dir / f"checkpoint_step_{resume_step}" if resume_step else base_dir
    )
    model, tokenizer, device = load_model_and_tokenizer(
        str(checkpoint_source), device=cfg["model"].get("device", "auto")
    )
    reference_rows = select_update_rows(splits, "forget", seq_len, args.order, seed)
    if args.condition == "forget":
        update_rows = reference_rows
        update_token_counts = count_token_lengths(
            reference_rows,
            tokenizer,
            int(cfg["model"].get("max_length", 256)),
            cfg["evaluation"].get("prompt_template", "{text}"),
        )
        reference_token_counts = list(update_token_counts)
    else:
        original_reference_rows = select_update_rows(
            splits, "forget", seq_len, "original", seed
        )
        update_rows, reference_token_counts, update_token_counts = (
            match_updates_to_reference_order(
                original_reference_rows,
                splits[UPDATE_SPLIT_BY_CONDITION[args.condition]],
                reference_rows,
                tokenizer,
                int(cfg["model"].get("max_length", 256)),
                cfg["evaluation"].get("prompt_template", "{text}"),
            )
        )
    update_manifest = [
        {
            "id": row["id"],
            "token_count": token_count,
            "paired_forget_id": reference_rows[index]["id"],
        }
        for index, (row, token_count) in enumerate(zip(update_rows, update_token_counts))
    ]
    update_info = {
        "update_examples": update_manifest,
        "update_token_counts": update_token_counts,
        "reference_forget_token_counts": reference_token_counts,
        "update_sequence_fingerprint": stable_fingerprint(
            {
                "updates": update_manifest,
                "reference_forget_token_counts": reference_token_counts,
            }
        ),
    }
    logger = setup_logger(run_dir)
    logger.info("Loaded shared base checkpoint on %s", device)
    run_condition(
        model,
        tokenizer,
        device,
        splits,
        update_rows,
        cfg,
        run_dir,
        args.condition,
        args.order,
        {
            **build_run_info(
                cfg,
                splits,
                device,
                manifest["resolved_revision"],
                manifest["checkpoint_sha256"],
            ),
            **update_info,
        },
        unlearner=build_unlearner(cfg["unlearning"].get("method", "ga")),
        resume_loader=lambda path: load_model_and_tokenizer(
            path, device=cfg["model"].get("device", "auto")
        ),
        restored_checkpoint_step=resume_step,
        logger=logger,
    )
    logger.info("Wrote metrics and checkpoints under %s", run_dir)

if __name__ == "__main__":
    main()
