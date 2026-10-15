import torch


@torch.no_grad()
def mean_hidden_layers(
    model, tokenizer, texts: list[str], device: str, max_length: int = 256,
    hidden_layers: tuple[int, ...] = (-1,),
) -> dict[int, torch.Tensor]:
    """Mask-aware mean-pooled hidden states for several layers from one forward pass."""
    model.eval()
    batch = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch, output_hidden_states=True, use_cache=False)
    mask = batch["attention_mask"].unsqueeze(-1)
    pooled = {}
    for layer in hidden_layers:
        hidden = outputs.hidden_states[layer]  # [batch, seq, hidden]
        layer_mask = mask.to(hidden.dtype)
        vectors = (hidden * layer_mask).sum(dim=1) / layer_mask.sum(dim=1).clamp_min(1)
        pooled[layer] = vectors.detach().float().cpu()
    return pooled


@torch.no_grad()
def mean_last_hidden(
    model, tokenizer, texts: list[str], device: str, max_length: int = 256,
    hidden_layer: int = -1,
) -> torch.Tensor:
    return mean_hidden_layers(
        model, tokenizer, texts, device, max_length=max_length, hidden_layers=(hidden_layer,)
    )[hidden_layer]
