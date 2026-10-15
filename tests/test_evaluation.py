import torch

import se_unlearning.evaluation.evaluate_checkpoint as checkpoint_eval
from se_unlearning.evaluation.evaluate_checkpoint import evaluate

FORGET = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
RETAIN = torch.tensor([[1.0, 0.0]])
CONTROL = torch.tensor([[0.0, 1.0]])


def patch(monkeypatch, captured):
    def fake_losses(model, tokenizer, rows, device, max_length):
        captured.setdefault("loss_prompts", []).append([row["text"] for row in rows])
        return [float(len(rows))] * len(rows)

    def fake_layers(model, tokenizer, selected, device, max_length, hidden_layers):
        captured["selected"] = selected
        captured["hidden_layers"] = list(hidden_layers)
        by_split = {"forget": FORGET, "retain": RETAIN, "control": CONTROL}
        return {
            name: {layer: by_split[name] * (1 + index) for index, layer in enumerate(hidden_layers)}
            for name in selected
        }

    monkeypatch.setattr(checkpoint_eval, "token_losses", fake_losses)
    monkeypatch.setattr(checkpoint_eval, "paired_representation_layers", fake_layers)


def test_evaluation_honors_example_limit_and_representation_layer(monkeypatch):
    splits = {
        "forget": [{"id": "f1", "text": "f1"}, {"id": "f2", "text": "f2"}, {"id": "f3", "text": "f3"}],
        "retain": [{"text": "r1"}, {"text": "r2"}, {"text": "r3"}],
        "control": [{"text": "c1"}, {"text": "c2"}, {"text": "c3"}],
    }
    captured = {}
    patch(monkeypatch, captured)

    metrics = evaluate(
        object(),
        object(),
        splits,
        "cpu",
        fixed_layer=-2,
        max_examples_per_split=2,
        prompt_template="Question: {text}",
    )

    assert captured["hidden_layers"] == [-2]
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
    assert metrics["per_example"]["forget"] == {"f1": 2.0, "f2": 2.0}
    assert "forget_update_mean_token_loss" not in metrics


def test_evaluation_adds_update_losses_and_extra_layers(monkeypatch):
    splits = {
        "forget": [{"id": "f1", "text": "f1"}, {"id": "f2", "text": "f2"}],
        "retain": [{"id": "r1", "text": "r1"}],
        "control": [{"id": "c1", "text": "c1"}],
        "forget_update": [{"id": "u1", "text": "u1"}],
    }
    captured = {}
    patch(monkeypatch, captured)

    metrics = evaluate(object(), object(), splits, "cpu", fixed_layer=-1, extra_layers=(6, -1, 12))

    assert captured["hidden_layers"] == [-1, 6, 12]
    assert set(captured["selected"]) == {"forget", "retain", "control"}
    assert metrics["forget_update_mean_token_loss"] == 1.0
    assert metrics["per_example"]["forget_update"] == {"u1": 1.0}
    for key in ("cosine_separation_L6", "cosine_separation_forget_retain_L12"):
        assert key in metrics
    assert "cosine_separation_L-1" not in metrics
