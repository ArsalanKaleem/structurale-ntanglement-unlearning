import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from se_unlearning.models.load_model import load_model_and_tokenizer
from se_unlearning.models.checkpoints import (
    save_base_checkpoint,
    validate_base_checkpoint,
    validate_model_revision,
)


def test_local_checkpoint_and_tokenizer_round_trip(tmp_path):
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            eos_token_id=vocabulary["[EOS]"],
            pad_token_id=vocabulary["[PAD]"],
        )
    )
    checkpoint = tmp_path / "checkpoint"
    manifest = save_base_checkpoint(
        model, tokenizer, checkpoint, "synthetic/tiny", "fixed-test-revision"
    )
    assert manifest["resolved_revision"] == "fixed-test-revision"
    assert len(manifest["checkpoint_sha256"]) == 64
    assert validate_base_checkpoint(
        checkpoint, "synthetic/tiny", "fixed-test-revision"
    ) == manifest

    reloaded, reloaded_tokenizer, device = load_model_and_tokenizer(
        str(checkpoint), device="cpu"
    )

    assert device == "cpu"
    assert reloaded.config.pad_token_id == vocabulary["[PAD]"]
    assert reloaded_tokenizer("hello")["input_ids"] == [vocabulary["hello"]]
    model.eval()
    reloaded.eval()
    inputs = torch.tensor([[vocabulary["hello"], vocabulary["[EOS]"]]])
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=inputs).logits, reloaded(input_ids=inputs).logits)


def test_base_checkpoint_rejects_different_model_revision(tmp_path):
    checkpoint = tmp_path / "missing"
    checkpoint.mkdir()
    (checkpoint / "base_manifest.json").write_text(
        '{"model_name":"model","requested_revision":"one","resolved_revision":"sha",'
        '"checkpoint_sha256":"abc"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_base_checkpoint(checkpoint, "model", "two")


def test_remote_model_requires_full_commit_sha():
    import pytest

    with pytest.raises(ValueError, match="40-character commit SHA"):
        validate_model_revision("publisher/model", "main")
    validate_model_revision("publisher/model", "a" * 40)


def test_base_checkpoint_rejects_content_changed_after_save(tmp_path):
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=4, n_positions=8, n_embd=8, n_layer=1, n_head=2)
    )
    checkpoint = tmp_path / "base"
    save_base_checkpoint(model, tokenizer, checkpoint, "model", "revision")
    weights = checkpoint / "model.safetensors"
    if not weights.exists():
        weights = checkpoint / "pytorch_model.bin"
    with weights.open("ab") as target:
        target.write(b"changed")

    with pytest.raises(ValueError, match="contents do not match"):
        validate_base_checkpoint(checkpoint, "model", "revision")
