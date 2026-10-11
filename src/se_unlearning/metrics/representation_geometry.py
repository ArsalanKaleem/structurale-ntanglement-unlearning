import torch
import torch.nn.functional as F

def cosine_separation(forget_vectors: torch.Tensor, control_vectors: torch.Tensor) -> float:
    '''Difference between mean within-forget similarity and mean cross-group similarity.

    Larger values indicate the forget representations are more internally similar than
    they are similar to controls. This is a descriptive proxy, not a validated measure
    of knowledge entanglement.
    '''
    if forget_vectors.ndim != 2 or control_vectors.ndim != 2:
        raise ValueError("Inputs must be 2-D [examples, hidden_size] tensors.")
    if forget_vectors.shape[1] != control_vectors.shape[1]:
        raise ValueError("Forget and control vectors must have the same feature dimension.")
    if len(forget_vectors) < 2 or len(control_vectors) < 1:
        raise ValueError("Need >=2 forget vectors and >=1 control vector.")
    f = F.normalize(forget_vectors.float(), dim=-1)
    c = F.normalize(control_vectors.float(), dim=-1)
    within = f @ f.T
    mask = ~torch.eye(len(f), dtype=torch.bool, device=within.device)
    within_mean = within[mask].mean().item()
    cross_mean = (f @ c.T).mean().item()
    return float(within_mean - cross_mean)

def paired_representation_vectors(
    model, tokenizer, splits, device, max_length=256, hidden_layer=-1
):
    from se_unlearning.models.activations import mean_last_hidden
    result = {}
    for name, rows in splits.items():
        result[name] = mean_last_hidden(
            model,
            tokenizer,
            [r["text"] for r in rows],
            device,
            max_length=max_length,
            hidden_layer=hidden_layer,
        )
    return result
