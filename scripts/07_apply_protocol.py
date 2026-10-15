"""Apply a frozen protocol inversion rule (v1.0 or v2.0) to 06_analyze.py output.

v2.0 (docs/protocol_v2.md) adds two requirements to v1.0's rule below: a step counts only
if every stream keeps >= 0.8 of its learned non-target knowledge (collapse guard), and a
reliable step must keep its sign and |mean| >= tau in every leave-one-seed-out subset.

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


def apply_rule(
    analysis: dict, baseline: float, order: str = PRIMARY_ORDER, horizon: int = HORIZON
) -> dict:
    tau = tolerance(baseline)
    per_reference = {}
    for reference in ("control", "retain", "no_update"):
        summaries = sorted(
            (
                s for s in analysis["summaries"]
                if s["order"] == order and s["control_condition"] == reference
                and 1 <= s["step"] <= horizon
            ),
            key=lambda s: s["step"],
        )
        steps = [s["step"] for s in summaries]
        if steps != list(range(1, horizon + 1)):
            raise ValueError(f"{reference}: expected steps 1-{horizon}, found {steps}")
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


GUARD_MIN_KNOWLEDGE = 0.8
NON_TARGET_SPLITS = {
    "forget": ("retain", "control"),
    "control": ("forget", "retain"),
    "retain": ("forget", "control"),
}


def guarded_steps(
    runs_root: Path, pre_losses: dict, order: str = PRIMARY_ORDER, horizon: int = HORIZON
) -> dict:
    """Mean (over seeds) knowledge fraction of each stream's non-target splits per step."""
    knowledge = {}
    for condition, splits in NON_TARGET_SPLITS.items():
        per_seed = []
        for metrics_path in sorted(runs_root.glob(f"seed_*/condition_{condition}_{order}/metrics.json")):
            rows = json.loads(metrics_path.read_text(encoding="utf-8"))
            per_seed.append([
                min(
                    (pre_losses[split] - row[f"{split}_mean_token_loss"])
                    / (pre_losses[split] - rows[0][f"{split}_mean_token_loss"])
                    for split in splits
                )
                for row in rows
            ])
        if not per_seed:
            raise FileNotFoundError(f"No {condition} {order} runs under {runs_root}")
        knowledge[condition] = [sum(col) / len(col) for col in zip(*per_seed)]
    return {
        "min_knowledge_by_stream": knowledge,
        "valid_steps": [
            t for t in range(1, horizon + 1)
            if all(k[t] >= GUARD_MIN_KNOWLEDGE for k in knowledge.values())
        ],
    }


def leave_one_out_stable(values: list[float], tau: float, sign: int) -> bool:
    if len(values) < 3:
        return False
    for drop in range(len(values)):
        rest = values[:drop] + values[drop + 1:]
        if sign * (sum(rest) / len(rest)) < tau:
            return False
    return True


def apply_rule_v2(
    analysis: dict, baseline: float, guard: dict, order: str = PRIMARY_ORDER,
    horizon: int = HORIZON,
) -> dict:
    result = apply_rule(analysis, baseline, order, horizon)
    tau = result["tau"]
    valid = set(guard["valid_steps"])
    steps = list(range(1, horizon + 1))
    for reference, info in result["references"].items():
        signs = []
        for step in steps:
            sign = info["step_classification"][step]
            values = [
                row["paired_effect"] for row in analysis["per_seed_effects"]
                if row["order"] == order and row["control_condition"] == reference
                and row["step"] == step
            ]
            if step not in valid or (sign and not leave_one_out_stable(values, tau, sign)):
                sign = 0
            signs.append(sign)
        info["v2_step_classification"] = dict(zip(steps, signs))
        info["v2_reliable_positive_steps"] = [t for t, v in zip(steps, signs) if v == 1]
        info["v2_reliable_negative_steps"] = [t for t, v in zip(steps, signs) if v == -1]
        info["v2_inversion_patterns"] = sorted(inversion_patterns(signs))
    control_patterns = set(result["references"]["control"]["v2_inversion_patterns"])
    retain_patterns = set(result["references"]["retain"]["v2_inversion_patterns"])
    if control_patterns & retain_patterns:
        verdict = "supported"
    elif control_patterns:
        verdict = "partial (inversion relative to unrelated unlearning only)"
    else:
        verdict = "not detected"
    result["v1_style_verdict"] = result.pop("verdict")
    result["verdict"] = verdict
    result["guard"] = guard
    result["n_valid_steps"] = len(valid)
    return result


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
    p.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    p.add_argument("--finetune-report", help="v2: fine-tune report with pre-fine-tuning losses")
    p.add_argument("--dataset-name", help="v2: main dataset file name in the report")
    p.add_argument("--primary-metric", default=PRIMARY_METRIC)
    p.add_argument("--horizon", type=int, default=HORIZON, help="Number of requests analyzed")
    args = p.parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    metric = analysis.get("metric", PRIMARY_METRIC)
    baseline = shared_baseline(Path(args.runs), metric)
    if args.protocol == "v2":
        if not args.finetune_report or not args.dataset_name:
            p.error("--protocol v2 needs --finetune-report and --dataset-name")
        report = json.loads(Path(args.finetune_report).read_text(encoding="utf-8"))
        guard = guarded_steps(
            Path(args.runs), report["loss_before"][args.dataset_name], horizon=args.horizon
        )
        result = apply_rule_v2(analysis, baseline, guard, horizon=args.horizon)
    else:
        result = apply_rule(analysis, baseline, horizon=args.horizon)
    result["protocol"] = args.protocol
    result["confirmatory"] = metric == args.primary_metric
    result["order_sensitivity"] = order_sensitivity(analysis, step=args.horizon)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
