import torch

@torch.no_grad()
def mean_last_hidden(
    model, tokenizer, texts: list[str], device: str, max_length: int = 256,
    hidden_layer: int = -1,
) -> torch.Tensor:
    model.eval()
    batch = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[hidden_layer]  # [batch, seq, hidden]
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.detach().float().cpu()
