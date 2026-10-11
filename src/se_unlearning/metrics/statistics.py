import numpy as np


def paired_effect_summary(
    effects: list[float], seed: int, bootstrap_samples: int = 5000
) -> dict:
    if not effects:
        raise ValueError("At least one paired effect is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    values = np.asarray(effects, dtype=float)
    mean_effect = float(values.mean())
    if len(values) < 2:
        confidence_interval = None
        standardized_effect = None
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        bootstrap_means = values[indices].mean(axis=1)
        confidence_interval = [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ]
        spread = float(values.std(ddof=1))
        standardized_effect = mean_effect / spread if spread > 0 else None
    return {
        "n_seeds": len(values),
        "mean_paired_effect": mean_effect,
        "standardized_paired_effect_dz": standardized_effect,
        "bootstrap_95_percentile_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples if len(values) >= 2 else 0,
    }
