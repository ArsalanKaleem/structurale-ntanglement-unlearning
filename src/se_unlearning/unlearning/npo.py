import torch
import torch.nn.functional as F

from se_unlearning.unlearning.base import UnlearningMethod


def _encode(tokenizer, texts, device, max_length):
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return {k: v.to(device) for k, v in enc.items()}


def _sequence_logprobs(model, enc) -> torch.Tensor:
    """Sum of next-token log-probabilities per row, ignoring padding."""
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].to(logits.dtype)
    token_logprobs = torch.gather(
        F.log_softmax(logits.float(), dim=-1), 2, labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logprobs * mask).sum(dim=1)


@torch.no_grad()
def reference_logprobs(model, tokenizer, texts: list[str], device: str, max_length: int = 256) -> list[float]:
    """Sequence log-probabilities under the reference (pre-unlearning) model, one text at a time."""
    model.eval()
    return [
        float(_sequence_logprobs(model, _encode(tokenizer, [text], device, max_length))[0])
        for text in texts
    ]


class NPOUnlearner(UnlearningMethod):
    '''Negative Preference Optimization (Zhang et al., 2024) on the current request.

    loss = -(2 / beta) * log sigmoid(-beta * (log p_theta(y) - log p_ref(y))), with sequence
    log-probabilities and p_ref the shared pre-unlearning checkpoint. Each row must carry a
    precomputed ``reference_logprob``. Unlike gradient ascent the objective is bounded:
    its gradient vanishes as the sequence becomes unlikely.
    '''

    def update(self, model, tokenizer, rows, device, config, logger=None):
        model.train()
        lr = float(config["learning_rate"])
        steps = int(config["max_steps_per_request"])
        if steps < 1:
            raise ValueError("max_steps_per_request must be >= 1")
        beta = float(config.get("npo_beta", 0.1))
        if beta <= 0:
            raise ValueError("npo_beta must be > 0")
        batch_size = int(config.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if int(config.get("gradient_accumulation_steps", 1)) != 1:
            raise ValueError("gradient_accumulation_steps other than 1 is not implemented")
        if not rows:
            raise ValueError("Cannot unlearn an empty request.")
        missing = [r.get("id") for r in rows if "reference_logprob" not in r]
        if missing:
            raise ValueError(f"NPO rows need a precomputed reference_logprob: {missing}")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        step_losses = []
        for step in range(steps):
            start = (step * batch_size) % len(rows)
            batch = rows[start:start + batch_size] or rows[:batch_size]
            enc = _encode(
                tokenizer, [r["text"] for r in batch], device, int(config.get("max_length", 256))
            )
            logprob = _sequence_logprobs(model, enc)
            reference = torch.tensor(
                [float(r["reference_logprob"]) for r in batch], device=logprob.device
            )
            loss = -(2.0 / beta) * F.logsigmoid(-beta * (logprob - reference)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_losses.append(float(loss.detach().cpu()))
        model.eval()
        if logger:
            logger.info("NPO update: rows=%d steps=%d mean_npo_loss=%.4f",
                        len(rows), steps, sum(step_losses) / len(step_losses))
        return {"steps": steps, "mean_npo_loss": sum(step_losses) / len(step_losses)}
