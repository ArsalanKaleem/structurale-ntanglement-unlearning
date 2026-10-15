import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from se_unlearning.unlearning.gradient_ascent import build_unlearner
from se_unlearning.unlearning.npo import NPOUnlearner, reference_logprobs

VOCABULARY = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "alpha": 3, "beta": 4, "gamma": 5}


def tiny_model_and_tokenizer():
    torch.manual_seed(0)
    backend = Tokenizer(WordLevel(VOCABULARY, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=len(VOCABULARY), n_positions=8, n_embd=8, n_layer=1, n_head=2)
    )
    return model, tokenizer


def config(**overrides):
    return {"learning_rate": 0.05, "max_steps_per_request": 5, "max_length": 8, **overrides}


def test_npo_lowers_the_forget_sequence_probability():
    model, tokenizer = tiny_model_and_tokenizer()
    text = "alpha beta gamma"
    before = reference_logprobs(model, tokenizer, [text], "cpu")[0]
    row = {"id": "f", "text": text, "reference_logprob": before}
    NPOUnlearner().update(model, tokenizer, [row], "cpu", config())
    after = reference_logprobs(model, tokenizer, [text], "cpu")[0]
    assert after < before


def test_npo_initial_loss_matches_closed_form():
    model, tokenizer = tiny_model_and_tokenizer()
    text = "alpha beta"
    row = {"id": "f", "text": text, "reference_logprob": reference_logprobs(model, tokenizer, [text], "cpu")[0]}
    result = NPOUnlearner().update(
        model, tokenizer, [row], "cpu", config(learning_rate=0.0, max_steps_per_request=1, npo_beta=0.1)
    )
    assert result["mean_npo_loss"] == pytest.approx((2 / 0.1) * math.log(2), rel=1e-4)


def test_npo_requires_reference_logprobs():
    model, tokenizer = tiny_model_and_tokenizer()
    with pytest.raises(ValueError, match="reference_logprob"):
        NPOUnlearner().update(model, tokenizer, [{"id": "f", "text": "alpha"}], "cpu", config())


def test_reference_logprobs_match_manual_sum():
    model, tokenizer = tiny_model_and_tokenizer()
    model.eval()
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    manual = torch.log_softmax(logits[0, :-1], dim=-1).gather(1, ids[0, 1:].unsqueeze(1)).sum()
    assert reference_logprobs(model, tokenizer, ["alpha beta gamma"], "cpu")[0] == pytest.approx(
        float(manual), abs=1e-5
    )


def test_build_unlearner_knows_npo():
    assert isinstance(build_unlearner("npo"), NPOUnlearner)
    with pytest.raises(ValueError, match="not implemented"):
        build_unlearner("rmu")
