from types import SimpleNamespace

import pytest
import torch

from se_unlearning.metrics.representation_geometry import cosine_separation
from se_unlearning.models.activations import mean_last_hidden

def test_cosine_separation_finite():
    f = torch.tensor([[1., 0.], [0.9, 0.1], [0.8, 0.2]])
    c = torch.tensor([[0., 1.], [0.1, 0.9]])
    score = cosine_separation(f, c)
    assert isinstance(score, float)
    assert -2 <= score <= 2

def test_cosine_separation_requires_two_forget_vectors():
    with pytest.raises(ValueError):
        cosine_separation(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]))

def test_cosine_separation_matches_hand_calculation():
    forget = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    control = torch.tensor([[1.0, 0.0]])

    assert cosine_separation(forget, control) == pytest.approx(-0.5)

def test_representation_uses_configured_hidden_layer():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class Model:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(
                hidden_states=(
                    torch.ones((1, 2, 3)),
                    torch.full((1, 2, 3), 4.0),
                )
            )

    vector = mean_last_hidden(Model(), Tokenizer(), ["example"], "cpu", hidden_layer=0)

    torch.testing.assert_close(vector, torch.ones((1, 3)))
