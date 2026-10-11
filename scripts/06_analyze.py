import argparse
import csv
import json
from pathlib import Path

import numpy as np

from se_unlearning.metrics.statistics import paired_effect_summary


def _load_run(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metrics_path = path.with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Run is not complete: {path.parent} ({metadata.get('status')})")
    by_step = {int(row["step"]): row for row in metrics}
    if sorted(by_step) != list(range(len(by_step))):
        raise ValueError(f"Run metrics have missing or duplicate steps: {metrics_path}")
    return {"metadata": metadata, "metrics": by_step}


def _same_baseline(runs: list[dict]) -> None:
    baselines = [
        run["metrics"][0]["cosine_separation"]
        for run in runs
        if 0 in run["metrics"]
    ]
    if len(baselines) != len(runs) or not np.allclose(
        baselines, baselines[0], rtol=0.0, atol=1e-6
    ):
        raise ValueError("Conditions do not share the same step-0 structural baseline")


def analyze(input_dir: Path, output_dir: Path, bootstrap_samples: int = 5000) -> list[dict]:
    runs = []
    for metadata_path in input_dir.rglob("condition_*_*/run.json"):
        run = _load_run(metadata_path)
        run["path"] = metadata_path.parent
        runs.append(run)
    if not runs:
        raise FileNotFoundError(f"No condition run metadata found under {input_dir}")

    treatment_runs = [
        run for run in runs if run["metadata"].get("condition") == "forget"
    ]
    outputs = []
    for treatment in treatment_runs:
        meta = treatment["metadata"]
        order = meta["order"]
        seed = int(meta["seed"])
        paired = {"no_update": [], "control": [], "retain": []}
        for condition in paired:
            matches = [
                candidate
                for candidate in runs
                if candidate["metadata"].get("condition") == condition
                and candidate["metadata"].get("order") == order
                and int(candidate["metadata"].get("seed", -1)) == seed
            ]
            if condition == "no_update":
                matches = [
                    candidate
                    for candidate in runs
                    if candidate["metadata"].get("condition") == "no_update"
                    and int(candidate["metadata"].get("seed", -1)) == seed
                ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {condition!r} match for seed={seed}, order={order}; "
                    f"found {len(matches)}"
                )
            reference = matches[0]
            for key in (
                "dataset_sha256",
                "resolved_model_revision",
                "base_checkpoint_sha256",
                "evaluation_fingerprint",
                "hyperparameters",
                "evaluation_settings",
            ):
                if meta.get(key) != reference["metadata"].get(key):
                    raise ValueError(f"{condition} run mismatches treatment {key}")
            paired[condition].append(reference)

        _same_baseline([treatment, *paired["no_update"], *paired["control"], *paired["retain"]])
        for control_condition in ("no_update", "control", "retain"):
            control_run = paired[control_condition][0]
            if control_condition == "no_update":
                if set(control_run["metrics"]) != {0}:
                    raise ValueError("No-update baseline must contain only step 0")
                common_steps = sorted(treatment["metrics"])
            else:
                common_steps = sorted(
                    set(treatment["metrics"]) & set(control_run["metrics"])
                )
                if not common_steps or common_steps[0] != 0:
                    raise ValueError(f"Missing matched metric steps for {control_condition}")
                if max(treatment["metrics"]) != max(control_run["metrics"]):
                    raise ValueError(f"Update counts differ for {control_condition}")
                if treatment["metadata"].get("update_count") != control_run[
                    "metadata"
                ].get("update_count"):
                    raise ValueError(f"Recorded update counts differ for {control_condition}")
                if treatment["metadata"].get("optimizer_step_count") != control_run[
                    "metadata"
                ].get("optimizer_step_count"):
                    raise ValueError(f"Optimizer-step counts differ for {control_condition}")
                if treatment["metadata"].get(
                    "reference_forget_token_counts"
                ) != control_run["metadata"].get("reference_forget_token_counts"):
                    raise ValueError(
                        f"Forget reference token lengths differ for {control_condition}"
                    )
            for step in common_steps[1:]:
                treatment_change = (
                    treatment["metrics"][step]["cosine_separation"]
                    - treatment["metrics"][0]["cosine_separation"]
                )
                reference_change = (
                    0.0
                    if control_condition == "no_update"
                    else (
                        control_run["metrics"][step]["cosine_separation"]
                        - control_run["metrics"][0]["cosine_separation"]
                    )
                )
                outputs.append(
                    {
                        "seed": seed,
                        "order": order,
                        "step": step,
                        "control_condition": control_condition,
                        "paired_effect": treatment_change - reference_change,
                    }
                )

    summaries = []
    groups = sorted(
        {
            (row["order"], row["step"], row["control_condition"])
            for row in outputs
        }
    )
    for order, step, control_condition in groups:
        values = [
            row["paired_effect"]
            for row in outputs
            if (row["order"], row["step"], row["control_condition"])
            == (order, step, control_condition)
        ]
        summaries.append(
            {
                "order": order,
                "step": step,
                "control_condition": control_condition,
                **paired_effect_summary(values, seed=step, bootstrap_samples=bootstrap_samples),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "primary_metric": "cosine_separation",
                "estimand": (
                    "paired difference in change from step 0: forget GA minus each "
                    "no-update, control GA, or retain GA reference"
                ),
                "interpretation": (
                    "Exploratory effect estimates with seed-level bootstrap intervals; "
                    "not evidence of causal inversion by themselves."
                ),
                "summaries": summaries,
                "per_seed_effects": outputs,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if summaries:
        with open(output_dir / "analysis.csv", "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Summarize paired seed-level structural metric effects and uncertainty."
    )
    parser.add_argument("--input", required=True, help="Root directory containing condition runs")
    parser.add_argument("--output", required=True, help="Directory for analysis.json and CSV")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    results = analyze(Path(args.input), Path(args.output), args.bootstrap_samples)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
