import argparse, json
from pathlib import Path
from se_unlearning.utils.config import load_config
from se_unlearning.utils.seeds import set_seed
from se_unlearning.utils.logging import setup_logger, write_run_metadata
from se_unlearning.data.load_dataset import load_jsonl, select_splits
from se_unlearning.models.load_model import load_model_and_tokenizer
from se_unlearning.evaluation.evaluate_checkpoint import evaluate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    out = Path(cfg["experiment"]["output_dir"])
    logger = setup_logger(out)
    write_run_metadata(out, cfg)
    rows = load_jsonl(cfg["data"]["input_jsonl"])
    limits = {
        "forget": cfg["data"].get("max_forget_examples"),
        "retain": cfg["data"].get("max_retain_examples"),
        "control": cfg["data"].get("max_control_examples"),
    }
    splits = select_splits(rows, limits)
    model, tokenizer, device = load_model_and_tokenizer(
        cfg["model"]["name_or_path"], cfg["model"].get("revision", "main"),
        cfg["model"].get("device", "auto")
    )
    logger.info("Loaded %s on %s", cfg["model"]["name_or_path"], device)
    metrics = evaluate(
        model,
        tokenizer,
        splits,
        device,
        int(cfg["model"].get("max_length", 256)),
        int(cfg["evaluation"].get("fixed_layer", -1)),
        int(cfg["evaluation"].get("max_eval_examples_per_split", 64)),
    )
    (out / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Baseline metrics: %s", metrics)

if __name__ == "__main__":
    main()
