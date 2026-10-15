import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, revision: str = "main", device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device
