import random

import torch


def finetune(model, tokenizer, texts: list[str], device: str, config: dict, seed: int) -> dict:
    """Full-parameter causal-LM fine-tuning so the model learns the fictional facts.

    Each epoch visits every text once in an order shuffled from ``seed + epoch``.
    """
    lr = float(config["learning_rate"])
    epochs = int(config["epochs"])
    batch_size = int(config.get("batch_size", 16))
    max_length = int(config.get("max_length", 256))
    if epochs < 1 or batch_size < 1:
        raise ValueError("finetune epochs and batch_size must be >= 1")
    if not texts:
        raise ValueError("No fine-tuning texts")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    per_epoch_loss = []
    steps = 0
    for epoch in range(epochs):
        order = list(texts)
        random.Random(seed + epoch).shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            enc = tokenizer(
                order[start:start + batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            loss = model(**enc, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach().cpu()))
            steps += 1
        per_epoch_loss.append(sum(epoch_losses) / len(epoch_losses))
    model.eval()
    return {"epochs": epochs, "optimizer_steps": steps, "per_epoch_train_loss": per_epoch_loss}
