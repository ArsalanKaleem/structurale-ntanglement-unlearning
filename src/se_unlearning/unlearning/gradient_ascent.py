import torch
from se_unlearning.unlearning.base import UnlearningMethod

class GradientAscentUnlearner(UnlearningMethod):
    '''Simple GA baseline: maximize next-token loss on the current forget request.

    This is a minimal research baseline, not a claim of equivalence to any paper-specific
    GA setup. Use small steps and evaluate retain utility to detect collateral damage.
    '''
    def update(self, model, tokenizer, rows, device, config, logger=None):
        model.train()
        lr = float(config["learning_rate"])
        steps = int(config["max_steps_per_request"])
        if steps < 1:
            raise ValueError("max_steps_per_request must be >= 1")
        batch_size = int(config.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
        if gradient_accumulation_steps != 1:
            raise ValueError("gradient_accumulation_steps other than 1 is not implemented")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        texts = [r["text"] for r in rows]
        if not texts:
            raise ValueError("Cannot unlearn an empty request.")
        step_losses = []
        for step in range(steps):
            start = (step * batch_size) % len(texts)
            batch_texts = texts[start:start + batch_size]
            if not batch_texts:
                batch_texts = texts[:batch_size]
            enc = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=int(config.get("max_length", 256))
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels, use_cache=False)
            loss = out.loss
            # Gradient ascent maximizes loss, so negate the objective before backward.
            (-loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_losses.append(float(loss.detach().cpu()))
        model.eval()
        if logger:
            logger.info("GA update: rows=%d steps=%d mean_forget_loss=%.4f",
                        len(rows), steps, sum(step_losses) / len(step_losses))
        return {"steps": steps, "mean_forget_loss": sum(step_losses) / len(step_losses)}

def build_unlearner(method: str):
    if method.lower() == "ga":
        return GradientAscentUnlearner()
    raise ValueError(
        f"Method {method!r} is not implemented in this starter. Use 'ga'; implement and test "
        "NPO/RMU against their original papers before reporting those methods."
    )
