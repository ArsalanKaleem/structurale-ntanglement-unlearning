"""Fine-tune the pinned model on the fictional facts and save the shared base checkpoint.

All rows of every listed dataset (main and calibration) are learned, so later unlearning
removes knowledge the model actually has. The checkpoint is written to
``experiment.shared_base_dir`` with a base manifest; a sibling ``*_report.json`` records
per-split losses before and after fine-tuning and the pre-registered knowledge gate:
post/pre mean token loss must be <= ``finetune.max_loss_ratio`` for every split.
"""

import argparse
import json
from pathlib import Path

from se_unlearning.data.load_dataset import SPLIT_ORDER, load_jsonl
from se_unlearning.metrics.forget_retain import mean_of_losses, token_losses
from se_unlearning.models.checkpoints import (
    save_base_checkpoint,
    validate_base_checkpoint,
    validate_model_revision,
)
from se_unlearning.models.finetune import finetune
from se_unlearning.models.load_model import load_model_and_tokenizer
from se_unlearning.utils.artifacts import sha256_file, write_json_atomic
from se_unlearning.utils.config import load_config
from se_unlearning.utils.seeds import set_seed


def split_losses(model, tokenizer, datasets, device, max_length):
    return {
        Path(path).name: {
            split: mean_of_losses(token_losses(
                model, tokenizer, [r for r in rows if r["split"] == split], device, max_length
            ))
            for split in SPLIT_ORDER
        }
        for path, rows in datasets.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ft = cfg["finetune"]
    out_dir = Path(cfg["experiment"]["shared_base_dir"])
    report_path = out_dir.with_name(out_dir.name + "_report.json")
    model_name = cfg["model"]["name_or_path"]
    revision = cfg["model"].get("revision", "main")
    if out_dir.exists() and report_path.exists():
        validate_base_checkpoint(out_dir, model_name, revision)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("Fine-tuned base already exists; gate passed:", report["knowledge_gate_passed"])
        return
    validate_model_revision(model_name, revision)
    seed = int(ft.get("seed", 0))
    set_seed(seed)
    datasets = {path: load_jsonl(path) for path in ft["datasets"]}
    texts = [row["text"] for rows in datasets.values() for row in rows]
    max_length = int(cfg["model"].get("max_length", 256))
    model, tokenizer, device = load_model_and_tokenizer(
        model_name, revision, cfg["model"].get("device", "auto")
    )
    before = split_losses(model, tokenizer, datasets, device, max_length)
    training = finetune(model, tokenizer, texts, device, {**ft, "max_length": max_length}, seed)
    after = split_losses(model, tokenizer, datasets, device, max_length)
    max_ratio = float(ft.get("max_loss_ratio", 0.3))
    ratios = {
        name: {split: after[name][split] / before[name][split] for split in SPLIT_ORDER}
        for name in before
    }
    gate = all(r <= max_ratio for per in ratios.values() for r in per.values())
    manifest = save_base_checkpoint(model, tokenizer, out_dir, model_name, revision)
    report = {
        "model_name": model_name,
        "model_revision": revision,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "datasets": {path: sha256_file(path) for path in ft["datasets"]},
        "n_texts": len(texts),
        "finetune_config": ft,
        "training": training,
        "loss_before": before,
        "loss_after": after,
        "loss_ratio_after_over_before": ratios,
        "max_loss_ratio": max_ratio,
        "knowledge_gate_passed": gate,
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2))
    if not gate:
        raise SystemExit("Knowledge gate failed: the model did not learn the facts; see report")


if __name__ == "__main__":
    main()
