import torch

import se_unlearning.evaluation.evaluate_checkpoint as checkpoint_eval
from se_unlearning.evaluation.evaluate_checkpoint import evaluate


def test_evaluation_honors_example_limit_and_representation_layer(monkeypatch):
    splits = {
        "forget": [{"text": "f1"}, {"text": "f2"}, {"text": "f3"}],
        "retain": [{"text": "r1"}, {"text": "r2"}, {"text": "r3"}],
        "control": [{"text": "c1"}, {"text": "c2"}, {"text": "c3"}],
    }
    captured = {}

    def fake_loss(model, tokenizer, rows, device, max_length):
        captured.setdefault("loss_prompts", []).append([row["text"] for row in rows])
        return float(len(rows))

    def fake_vectors(model, tokenizer, selected, device, max_length, hidden_layer):
        captured["selected"] = selected
        captured["vector_prompts"] = {
            name: [row["text"] for row in rows] for name, rows in selected.items()
        }
        captured["hidden_layer"] = hidden_layer
        return {
            "forget": torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
            "retain": torch.tensor([[1.0, 0.0]]),
            "control": torch.tensor([[0.0, 1.0]]),
        }

    monkeypatch.setattr(checkpoint_eval, "mean_token_loss", fake_loss)
    monkeypatch.setattr(checkpoint_eval, "paired_representation_vectors", fake_vectors)

    metrics = evaluate(
        object(),
        object(),
        splits,
        "cpu",
        fixed_layer=-2,
        max_examples_per_split=2,
        prompt_template="Question: {text}",
    )

    assert captured["hidden_layer"] == -2
    assert {name: len(rows) for name, rows in captured["selected"].items()} == {
        "forget": 2,
        "retain": 2,
        "control": 2,
    }
    assert metrics["forget_mean_token_loss"] == 2.0
    assert metrics["cosine_separation_forget_retain"] < metrics["cosine_separation"]
    assert captured["loss_prompts"][0] == [
        "Question: f1",
        "Question: f2",
    ]
