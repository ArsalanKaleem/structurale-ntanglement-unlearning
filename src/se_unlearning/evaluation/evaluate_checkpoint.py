from pathlib import Path
from se_unlearning.data.prompts import render_rows
from se_unlearning.metrics.forget_retain import mean_of_losses, token_losses
from se_unlearning.metrics.representation_geometry import paired_representation_layers
from se_unlearning.metrics.representation_geometry import cosine_separation
from se_unlearning.utils.artifacts import load_metric_rows, write_metric_rows

GEOMETRY_SPLITS = ("forget", "retain", "control")
# Loss-only split: direct efficacy on the update texts themselves.
LOSS_ONLY_SPLITS = ("forget_update",)


def evaluate(
    model, tokenizer, splits, device, max_length=256, fixed_layer=-1,
    max_examples_per_split=None, prompt_template="{text}", extra_layers=(),
):
    """Evaluate one checkpoint.

    Returns aggregate metrics plus a ``per_example`` mapping {split: {row id: loss}}
    that the runner stores separately for entity-level analyses.
    """
    if max_examples_per_split is not None and max_examples_per_split < 1:
        raise ValueError("max_examples_per_split must be >= 1")
    names = [*GEOMETRY_SPLITS, *(n for n in LOSS_ONLY_SPLITS if splits.get(n))]
    eval_splits = {
        name: render_rows(
            splits[name][:max_examples_per_split]
            if max_examples_per_split is not None else splits[name],
            prompt_template,
        )
        for name in names
    }
    metrics = {}
    per_example = {}
    for name, rows in eval_splits.items():
        losses = token_losses(model, tokenizer, rows, device, max_length=max_length)
        metrics[f"{name}_mean_token_loss"] = mean_of_losses(losses)
        per_example[name] = {
            str(row.get("id", index)): loss for index, (row, loss) in enumerate(zip(rows, losses))
        }
    layers = [fixed_layer, *(layer for layer in extra_layers if layer != fixed_layer)]
    vectors = paired_representation_layers(
        model,
        tokenizer,
        {name: eval_splits[name] for name in GEOMETRY_SPLITS},
        device,
        max_length=max_length,
        hidden_layers=layers,
    )
    for layer in layers:
        suffix = "" if layer == fixed_layer else f"_L{layer}"
        metrics[f"cosine_separation{suffix}"] = cosine_separation(
            vectors["forget"][layer], vectors["control"][layer]
        )
        # Same-domain entanglement between forget and retain rows.
        metrics[f"cosine_separation_forget_retain{suffix}"] = cosine_separation(
            vectors["forget"][layer], vectors["retain"][layer]
        )
    metrics["per_example"] = per_example
    return metrics

def append_metric(path: str | Path, row: dict):
    path = Path(path)
    rows = load_metric_rows(path.with_suffix(".json"))
    rows.append(row)
    write_metric_rows(path, rows)
