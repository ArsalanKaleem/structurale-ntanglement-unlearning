"""Entity-level spill-over of sequential unlearning (protocol v2, H2).

Uses per-example losses (per_example/step_N.json) from shuffled-order runs of one method.
Changes are loss increases from step 0 (post-fine-tuning), paired within seed.

- H2a, same-domain spill-over: at the final step, the retain-split loss increase under the
  forget stream minus the increase under the matched control stream (unlearning unrelated
  knowledge). Retain authors share the domain but are never targeted.
- H2b, entity locality: at step 8 of the forget stream, the mean loss increase on held-out
  facts of authors already targeted minus that of authors not yet targeted.

Both are summarized with seed-bootstrap 95% intervals (5000 resamples). Per-step
trajectories are written for exploration.
"""

import argparse
import json
from pathlib import Path

from se_unlearning.data.load_dataset import load_jsonl
from se_unlearning.metrics.statistics import paired_effect_summary

H2B_STEP = 8


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load_steps(run_dir: Path) -> list[dict]:
    steps = sorted(
        int(p.stem.removeprefix("step_")) for p in (run_dir / "per_example").glob("step_*.json")
    )
    if steps != list(range(len(steps))):
        raise ValueError(f"Missing per-example steps in {run_dir}")
    return [json.loads((run_dir / "per_example" / f"step_{t}.json").read_text()) for t in steps]


def forget_stream_effects(run_dir: Path, entity_of: dict) -> list[dict]:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    order = [entity_of[example["id"]] for example in meta["update_examples"]]
    per_step = load_steps(run_dir)
    start = per_step[0]["forget"]
    effects = []
    for t, losses in enumerate(per_step):
        targeted = set(order[:t])
        deltas = {
            row_id: (loss - start[row_id]) if loss is not None and start[row_id] is not None else None
            for row_id, loss in losses["forget"].items()
        }
        effects.append({
            "step": t,
            "targeted_entities": len(targeted),
            "targeted_delta": _mean(d for i, d in deltas.items() if entity_of[i] in targeted),
            "untargeted_delta": _mean(d for i, d in deltas.items() if entity_of[i] not in targeted),
        })
    return effects


def split_delta(run_dir: Path, split: str) -> list[float]:
    rows = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    key = f"{split}_mean_token_loss"
    return [row[key] - rows[0][key] for row in rows]


def analyze(
    method_root: Path, dataset: Path, bootstrap_samples: int = 5000, h2b_step: int = H2B_STEP
) -> dict:
    entity_of = {row["id"]: row["entity"] for row in load_jsonl(dataset)}
    seeds = sorted(p for p in method_root.glob("seed_*") if p.is_dir())
    if not seeds:
        raise FileNotFoundError(f"No seed directories under {method_root}")
    h2a, h2b, trajectories = [], [], []
    for seed_dir in seeds:
        forget_dir = seed_dir / "condition_forget_shuffled"
        control_dir = seed_dir / "condition_control_shuffled"
        retain_forget = split_delta(forget_dir, "retain")
        retain_control = split_delta(control_dir, "retain")
        h2a.append(retain_forget[-1] - retain_control[-1])
        effects = forget_stream_effects(forget_dir, entity_of)
        at = effects[h2b_step]
        h2b.append(at["targeted_delta"] - at["untargeted_delta"])
        trajectories.append({
            "seed": seed_dir.name,
            "forget_stream": effects,
            "retain_delta_forget_stream": retain_forget,
            "retain_delta_control_stream": retain_control,
            "forget_delta_control_stream": split_delta(control_dir, "forget"),
        })
    final_step = len(trajectories[0]["retain_delta_forget_stream"]) - 1
    return {
        "method_root": method_root.as_posix(),
        "n_seeds": len(seeds),
        "H2a_same_domain_spillover": {
            "definition": "retain loss increase at final step: forget stream minus control stream",
            "step": final_step,
            **paired_effect_summary(h2a, seed=final_step, bootstrap_samples=bootstrap_samples),
            "per_seed": h2a,
        },
        "H2b_entity_locality": {
            "definition": "held-out forget-fact loss increase: targeted minus not-yet-targeted authors",
            "step": h2b_step,
            **paired_effect_summary(h2b, seed=h2b_step, bootstrap_samples=bootstrap_samples),
            "per_seed": h2b,
        },
        "trajectories": trajectories,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="Method root containing seed_* directories")
    p.add_argument("--dataset", required=True, help="Main dataset JSONL (for entity labels)")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--h2b-step", type=int, default=H2B_STEP)
    args = p.parse_args()
    result = analyze(Path(args.input), Path(args.dataset), args.bootstrap_samples, args.h2b_step)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "trajectories"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
