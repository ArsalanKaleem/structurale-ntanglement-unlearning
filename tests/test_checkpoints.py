import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from se_unlearning.models.load_model import load_model_and_tokenizer


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
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

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
