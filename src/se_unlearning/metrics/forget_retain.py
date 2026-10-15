import torch
import torch.nn.functional as F


@torch.no_grad()
def token_losses(
    model, tokenizer, rows: list[dict], device: str, max_length: int = 256
) -> list[float | None]:
    """Mean next-token loss per row; None for rows with fewer than two tokens."""
    model.eval()
    losses: list[float | None] = []
    for row in rows:
        enc = tokenizer(
            row["text"], return_tensors="pt", truncation=True, max_length=max_length
        )
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 2:
            losses.append(None)
            continue
        attention = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).to(device)
        out = model(input_ids=ids, attention_mask=attention, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses.append(float(loss.item()))
    return losses


def mean_of_losses(losses: list[float | None]) -> float:
    valid = [loss for loss in losses if loss is not None]
    if not valid:
        raise ValueError("No examples with at least two tokens are available for loss evaluation")
    return float(sum(valid) / len(valid))


def mean_token_loss(model, tokenizer, rows: list[dict], device: str, max_length: int = 256) -> float:
    return mean_of_losses(token_losses(model, tokenizer, rows, device, max_length=max_length))
