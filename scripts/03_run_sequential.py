import argparse, json
from pathlib import Path
from se_unlearning.utils.config import load_config
from se_unlearning.utils.seeds import set_seed
from se_unlearning.utils.logging import setup_logger, write_run_metadata
from se_unlearning.data.load_dataset import load_jsonl, select_splits
from se_unlearning.models.load_model import load_model_and_tokenizer
from se_unlearning.unlearning.gradient_ascent import build_unlearner
from se_unlearning.evaluation.evaluate_checkpoint import evaluate, append_metric
from se_unlearning.evaluation.controls import shuffled_copy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--order", choices=["original", "shuffled"], default="original")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    out = Path(cfg["experiment"]["output_dir"])
    run_dir = out / f"sequential_{args.order}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    write_run_metadata(run_dir, cfg)
    rows = load_jsonl(cfg["data"]["input_jsonl"])
    limits = {
        "forget": cfg["data"].get("max_forget_examples"),
        "retain": cfg["data"].get("max_retain_examples"),
        "control": cfg["data"].get("max_control_examples"),
    }
    splits = select_splits(rows, limits)
    forget_rows = list(splits["forget"])
    if args.order == "shuffled":
        forget_rows = shuffled_copy(forget_rows, seed + 1)
    seq_len = min(int(cfg["unlearning"].get("sequence_length", len(forget_rows))), len(forget_rows))
    forget_rows = forget_rows[:seq_len]
    model, tokenizer, device = load_model_and_tokenizer(
        cfg["model"]["name_or_path"], cfg["model"].get("revision", "main"),
        cfg["model"].get("device", "auto")
    )
    unlearner = build_unlearner(cfg["unlearning"].get("method", "ga"))
    metrics_path = run_dir / "sequential_metrics.csv"
    max_length = int(cfg["model"].get("max_length", 256))
    fixed_layer = int(cfg["evaluation"].get("fixed_layer", -1))
    max_eval_examples = int(cfg["evaluation"].get("max_eval_examples_per_split", 64))
    base = evaluate(
        model, tokenizer, splits, device, max_length, fixed_layer, max_eval_examples
    )
    append_metric(metrics_path, {"step": 0, "request_id": "baseline", **base})
    for step, row in enumerate(forget_rows, start=1):
        update_cfg = dict(cfg["unlearning"])
        update_cfg["max_length"] = max_length
        result = unlearner.update(model, tokenizer, [row], device, update_cfg, logger)
        step_dir = run_dir / f"checkpoint_step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)
        metrics = evaluate(
            model, tokenizer, splits, device, max_length, fixed_layer, max_eval_examples
        )
        append_metric(metrics_path, {
            "step": step, "request_id": row.get("request_id", row["id"]),
            "update": json.dumps(result), **metrics
        })
        logger.info("Step %d metrics: %s", step, metrics)
    logger.info("Wrote metrics to %s", metrics_path)

if __name__ == "__main__":
    main()
