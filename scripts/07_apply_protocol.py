"""Apply the frozen protocol v1.0 inversion rule to 06_analyze.py output.

The rule (docs/protocol.md, "Inversion rule") is implemented exactly as written:
- tau = 0.10 * |S(0)| from the shared base-checkpoint value (0.001 if |S(0)| < 0.01);
- a step is reliably positive/negative when the mean paired effect over the
  shuffled-order seeds is >= +tau / <= -tau and its bootstrap 95% interval excludes 0;
- an inversion is a reliable step followed later by a reliable step of opposite sign;
- "supported" requires an inversion pattern against control GA that also appears
  against retain GA; "partial" is control GA only; otherwise "not detected".
"""

import argparse
import json
import statistics
from pathlib import Path

PRIMARY_METRIC = "cosine_separation"
PRIMARY_ORDER = "shuffled"
HORIZON = 16


def tolerance(baseline: float) -> float:
    return 0.001 if abs(baseline) < 0.01 else 0.10 * abs(baseline)


def classify_step(summary: dict, tau: float) -> int:
    interval = summary.get("bootstrap_95_percentile_interval")
    if interval is None:
        return 0
    mean = summary["mean_paired_effect"]
    if mean >= tau and interval[0] > 0:
        return 1
    if mean <= -tau and interval[1] < 0:
        return -1
    return 0


def inversion_patterns(signs: list[int]) -> set[str]:
    patterns = set()
    for i, first in enumerate(signs):
        for later in signs[i + 1:]:
            if first == 1 and later == -1:
                patterns.add("positive_then_negative")
            if first == -1 and later == 1:
                patterns.add("negative_then_positive")
    return patterns


def apply_rule(analysis: dict, baseline: float, order: str = PRIMARY_ORDER) -> dict:
    tau = tolerance(baseline)
    per_reference = {}
    for reference in ("control", "retain", "no_update"):
        summaries = sorted(
            (
                s for s in analysis["summaries"]
                if s["order"] == order and s["control_condition"] == reference
                and 1 <= s["step"] <= HORIZON
            ),
            key=lambda s: s["step"],
        )
        steps = [s["step"] for s in summaries]
        if steps != list(range(1, HORIZON + 1)):
            raise ValueError(f"{reference}: expected steps 1-{HORIZON}, found {steps}")
        signs = [classify_step(s, tau) for s in summaries]
        per_reference[reference] = {
            "n_seeds": sorted({s["n_seeds"] for s in summaries}),
            "step_classification": dict(zip(steps, signs)),
            "reliable_positive_steps": [t for t, v in zip(steps, signs) if v == 1],
            "reliable_negative_steps": [t for t, v in zip(steps, signs) if v == -1],
            "inversion_patterns": sorted(inversion_patterns(signs)),
        }
    control_patterns = set(per_reference["control"]["inversion_patterns"])
    retain_patterns = set(per_reference["retain"]["inversion_patterns"])
    if control_patterns & retain_patterns:
        verdict = "supported"
    elif control_patterns:
        verdict = "partial (inversion relative to unrelated updates only)"
    else:
        verdict = "not detected"
    return {
        "metric": analysis.get("metric", PRIMARY_METRIC),
        "order": order,
        "baseline_value": baseline,
        "tau": tau,
        "verdict": verdict,
        "references": per_reference,
    }


def order_sensitivity(analysis: dict, step: int = HORIZON) -> dict:
    shuffled = [
        row["paired_effect"] for row in analysis["per_seed_effects"]
        if row["order"] == "shuffled" and row["control_condition"] == "control"
        and row["step"] == step
    ]
    original = [
        row["paired_effect"] for row in analysis["per_seed_effects"]
        if row["order"] == "original" and row["control_condition"] == "control"
        and row["step"] == step
    ]
    return {
        "step": step,
        "shuffled_values": shuffled,
        "shuffled_sd": statistics.stdev(shuffled) if len(shuffled) > 1 else None,
        "shuffled_range": [min(shuffled), max(shuffled)] if shuffled else None,
        "original_order_values": original,
    }


def shared_baseline(runs_root: Path, metric: str) -> float:
    values = []
    for path in runs_root.rglob("condition_no_update_original/metrics.json"):
        values.append(json.loads(path.read_text(encoding="utf-8"))[0][metric])
    if not values:
        raise FileNotFoundError(f"No no-update baselines under {runs_root}")
    if max(values) - min(values) > 1e-6:
        raise ValueError(f"No-update baselines differ across seeds: {values}")
    return values[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, help="Root containing seed_*/condition_* runs")
    p.add_argument("--analysis", required=True, help="analysis*.json from 06_analyze.py")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    metric = analysis.get("metric", PRIMARY_METRIC)
    result = apply_rule(analysis, shared_baseline(Path(args.runs), metric))
    result["confirmatory"] = metric == PRIMARY_METRIC
    result["order_sensitivity"] = order_sensitivity(analysis)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
