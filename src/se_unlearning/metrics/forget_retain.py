import torch
import torch.nn.functional as F

@torch.no_grad()
def mean_token_loss(model, tokenizer, rows: list[dict], device: str, max_length: int = 256) -> float:
    model.eval()
    losses = []
    for row in rows:
        enc = tokenizer(
            row["text"], return_tensors="pt", truncation=True, max_length=max_length
        )
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        attention = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).to(device)
        out = model(input_ids=ids, attention_mask=attention, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses.append(float(loss.item()))
    if not losses:
        raise ValueError("No examples with at least two tokens are available for loss evaluation")
    return float(sum(losses) / len(losses))
