from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from se_unlearning.unlearning.gradient_ascent import GradientAscentUnlearner


class ToyTokenizer:
    def __call__(self, texts, **kwargs):
        sequences = {
            "forget": [0, 1],
            "short": [0, 1],
            "long": [0, 1, 1],
        }
        encoded = [sequences[text] for text in texts]
        width = max(map(len, encoded))
        input_ids = torch.tensor([seq + [0] * (width - len(seq)) for seq in encoded])
        attention_mask = torch.tensor(
            [[1] * len(seq) + [0] * (width - len(seq)) for seq in encoded]
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ToyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.2))
        self.last_labels = None

    def forward(self, input_ids, attention_mask, labels, use_cache):
        self.last_labels = labels.detach().clone()
        logits = torch.zeros((*input_ids.shape, 2), device=input_ids.device)
        logits[..., 0] = self.logit
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 2),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


def test_gradient_ascent_increases_forget_loss():
    model = ToyCausalLM()
    tokenizer = ToyTokenizer()
    encoded = tokenizer(["forget"])
    before = model(
        **encoded, labels=encoded["input_ids"], use_cache=False
    ).loss.item()

    GradientAscentUnlearner().update(
        model,
        tokenizer,
        [{"text": "forget"}],
        "cpu",
        {"learning_rate": 0.1, "max_steps_per_request": 1, "max_length": 8},
    )

    after = model(
        **encoded, labels=encoded["input_ids"], use_cache=False
    ).loss.item()
    assert after > before


def test_gradient_ascent_masks_padding_labels():
    model = ToyCausalLM()

    GradientAscentUnlearner().update(
        model,
        ToyTokenizer(),
        [{"text": "short"}, {"text": "long"}],
        "cpu",
        {
            "learning_rate": 0.1,
            "max_steps_per_request": 1,
            "batch_size": 2,
            "max_length": 8,
        },
    )

    assert torch.all(model.last_labels[0, 2:] == -100)


def test_gradient_ascent_requires_positive_step_count():
    with pytest.raises(ValueError, match="max_steps_per_request"):
        GradientAscentUnlearner().update(
            ToyCausalLM(),
            ToyTokenizer(),
            [{"text": "forget"}],
            "cpu",
            {"learning_rate": 0.1, "max_steps_per_request": 0},
        )


def test_gradient_ascent_rejects_unsupported_accumulation():
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        GradientAscentUnlearner().update(
            ToyCausalLM(),
            ToyTokenizer(),
            [{"text": "forget"}],
            "cpu",
            {
                "learning_rate": 0.1,
                "max_steps_per_request": 1,
                "gradient_accumulation_steps": 2,
            },
        )
