"""Select unlearning hyperparameters from calibration runs (protocol v2, frozen rule).

Knowledge fraction of a split at step t:
    K(t) = (L_pre - L_t) / (L_pre - L_0)
with L_pre the split's mean token loss before fine-tuning (from the fine-tune report),
L_0 the post-fine-tuning value (step 0 of the run), and L_t after t requests.
K = 1 means fully known, K = 0 means back to the pre-fine-tuning level.

For each method, a calibration configuration is admissible when
- guard:    K(retain, t) >= 0.8 and K(control, t) >= 0.8 at every step t, and
- efficacy: 1 - K(targeted forget_update rows, final step) >= 0.5,
and the admissible configuration with the highest efficacy is selected (ties: lower
learning rate, then fewer steps per request). If none is admissible, the guard-passing
configuration with the highest efficacy is reported with status "efficacy_not_met";
if no configuration passes the guard, status is "no_config_passes_guard".
"""

import argparse
import json
from pathlib import Path

GUARD_MIN_KNOWLEDGE = 0.8
MIN_EFFICACY = 0.5


def knowledge_fraction(pre: float, start: float, current: float) -> float:
    gap = pre - start
    if gap <= 0:
        raise ValueError("Split was not learned: pre-fine-tuning loss <= post-fine-tuning loss")
    return (pre - current) / gap


def evaluate_run(run_dir: Path, pre_losses: dict) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError(f"Calibration run is not complete: {run_dir}")
    rows = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    final = max(int(r["step"]) for r in rows)
    guard = {}
    for split in ("retain", "control"):
        key = f"{split}_mean_token_loss"
        guard[split] = min(
            knowledge_fraction(pre_losses[split], rows[0][key], row[key]) for row in rows
        )
    targeted = [example["id"] for example in meta["update_examples"]]
    start = json.loads((run_dir / "per_example" / "step_0.json").read_text())["forget_update"]
    end = json.loads((run_dir / "per_example" / f"step_{final}.json").read_text())["forget_update"]
    start_loss = sum(start[i] for i in targeted) / len(targeted)
    end_loss = sum(end[i] for i in targeted) / len(targeted)
    efficacy = 1 - knowledge_fraction(pre_losses["forget_update"], start_loss, end_loss)
    hp = meta["hyperparameters"]
    return {
        "run": run_dir.as_posix(),
        "method": hp.get("method", "ga"),
        "learning_rate": float(hp["learning_rate"]),
        "max_steps_per_request": int(hp["max_steps_per_request"]),
        "min_retain_knowledge": guard["retain"],
        "min_control_knowledge": guard["control"],
        "passes_guard": min(guard.values()) >= GUARD_MIN_KNOWLEDGE,
        "efficacy": efficacy,
        "passes_efficacy": efficacy >= MIN_EFFICACY,
    }


def select(candidates: list[dict]) -> dict:
    def rank(c):
        return (-c["efficacy"], c["learning_rate"], c["max_steps_per_request"])

    admissible = sorted((c for c in candidates if c["passes_guard"] and c["passes_efficacy"]), key=rank)
    if admissible:
        return {"status": "selected", "selected": admissible[0]}
    guarded = sorted((c for c in candidates if c["passes_guard"]), key=rank)
    if guarded:
        return {"status": "efficacy_not_met", "selected": guarded[0]}
    return {"status": "no_config_passes_guard", "selected": None}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, help="Root with calibration condition_forget_* runs")
    p.add_argument("--finetune-report", required=True)
    p.add_argument("--dataset-name", required=True, help="Calibration dataset file name in the report")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = json.loads(Path(args.finetune_report).read_text(encoding="utf-8"))
    pre = report["loss_before"][args.dataset_name]
    candidates = [
        evaluate_run(path.parent, pre)
        for path in sorted(Path(args.runs).rglob("condition_forget_*/run.json"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No calibration runs under {args.runs}")
    result = {
        "rule": {"guard_min_knowledge": GUARD_MIN_KNOWLEDGE, "min_efficacy": MIN_EFFICACY},
        "by_method": {
            method: {**select([c for c in candidates if c["method"] == method]),
                     "candidates": [c for c in candidates if c["method"] == method]}
            for method in sorted({c["method"] for c in candidates})
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
