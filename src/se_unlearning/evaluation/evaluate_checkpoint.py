from pathlib import Path
from se_unlearning.data.prompts import render_rows
from se_unlearning.metrics.forget_retain import mean_token_loss
from se_unlearning.metrics.representation_geometry import paired_representation_vectors
from se_unlearning.metrics.representation_geometry import cosine_separation
from se_unlearning.utils.artifacts import load_metric_rows, write_metric_rows

def evaluate(
    model, tokenizer, splits, device, max_length=256, fixed_layer=-1,
    max_examples_per_split=None, prompt_template="{text}",
):
    splits = {
        name: splits[name] for name in ("forget", "retain", "control")
    }
    if max_examples_per_split is not None and max_examples_per_split < 1:
        raise ValueError("max_examples_per_split must be >= 1")
    eval_splits = {
        name: rows[:max_examples_per_split] if max_examples_per_split is not None else rows
        for name, rows in splits.items()
    }
    eval_splits = {
        name: render_rows(rows, prompt_template) for name, rows in eval_splits.items()
    }
    losses = {}
    for name, rows in eval_splits.items():
        losses[f"{name}_mean_token_loss"] = mean_token_loss(
            model, tokenizer, rows, device, max_length=max_length
        )
    vectors = paired_representation_vectors(
        model,
        tokenizer,
        eval_splits,
        device,
        max_length=max_length,
        hidden_layer=fixed_layer,
    )
    losses["cosine_separation"] = cosine_separation(vectors["forget"], vectors["control"])
    # Secondary (exploratory): same-domain entanglement between forget and retain rows.
    losses["cosine_separation_forget_retain"] = cosine_separation(
        vectors["forget"], vectors["retain"]
    )
    return losses

def append_metric(path: str | Path, row: dict):
    path = Path(path)
    rows = load_metric_rows(path.with_suffix(".json"))
    rows.append(row)
    write_metric_rows(path, rows)
