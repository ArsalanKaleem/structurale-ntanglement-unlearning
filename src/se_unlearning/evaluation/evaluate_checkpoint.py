from pathlib import Path
import pandas as pd
from se_unlearning.metrics.forget_retain import mean_token_loss
from se_unlearning.metrics.representation_geometry import paired_representation_vectors, cosine_separation

def evaluate(
    model, tokenizer, splits, device, max_length=256, fixed_layer=-1,
    max_examples_per_split=None,
):
    if max_examples_per_split is not None and max_examples_per_split < 1:
        raise ValueError("max_examples_per_split must be >= 1")
    eval_splits = {
        name: rows[:max_examples_per_split] if max_examples_per_split is not None else rows
        for name, rows in splits.items()
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
    return losses

def append_metric(path: str | Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
