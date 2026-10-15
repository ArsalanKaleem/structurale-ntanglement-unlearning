"""Plot pilot trajectories: structural metric, paired effects, and collapse."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette, in fixed order.
COLORS = {"forget": "#2a78d6", "control": "#eb6834", "retain": "#1baf7a"}
LABELS = {"forget": "Forget GA", "control": "Control GA", "retain": "Retain GA"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_metrics(root: Path, seed: int, condition: str, order: str) -> list[dict]:
    path = root / f"seed_{seed}" / f"condition_{condition}_{order}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Completed forget requests", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def plot(runs: Path, analysis: dict, protocol: dict, metric: str, output: Path):
    seeds = sorted(int(p.name.removeprefix("seed_")) for p in runs.glob("seed_*"))
    steps = list(range(17))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="#fcfcfb")

    ax = axes[0]
    for condition in ("forget", "control", "retain"):
        per_seed = [[row[metric] for row in load_metrics(runs, s, condition, "shuffled")] for s in seeds]
        for values in per_seed:
            ax.plot(steps, values, color=COLORS[condition], linewidth=0.8, alpha=0.3)
        mean = [sum(col) / len(col) for col in zip(*per_seed)]
        ax.plot(steps, mean, color=COLORS[condition], linewidth=2, label=LABELS[condition])
    style(ax, f"A. {metric} (thin: seeds; thick: mean)", metric)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    tau = protocol["tau"]
    for reference in ("control", "retain"):
        rows = sorted(
            (s for s in analysis["summaries"]
             if s["order"] == "shuffled" and s["control_condition"] == reference),
            key=lambda s: s["step"],
        )
        x = [r["step"] for r in rows]
        mean = [r["mean_paired_effect"] for r in rows]
        low = [r["bootstrap_95_percentile_interval"][0] for r in rows]
        high = [r["bootstrap_95_percentile_interval"][1] for r in rows]
        ax.fill_between(x, low, high, color=COLORS[reference], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=COLORS[reference], linewidth=2, marker="o", markersize=3,
                label=f"Forget GA minus {LABELS[reference]}")
    for level in (tau, -tau):
        ax.axhline(level, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axhline(0, color=MUTED, linewidth=0.6)
    style(ax, f"B. Paired effect, 95% band (dashed ±τ={tau:.4f})",
          "Paired effect")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for condition in ("forget", "control", "retain"):
        per_seed = [[row["retain_mean_token_loss"] for row in load_metrics(runs, s, condition, "shuffled")]
                    for s in seeds]
        mean = [sum(col) / len(col) for col in zip(*per_seed)]
        ax.plot(steps, mean, color=COLORS[condition], linewidth=2, label=LABELS[condition])
    base = load_metrics(runs, seeds[0], "no_update", "original")[0]["retain_mean_token_loss"]
    ax.axhline(base, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    style(ax, "C. Retain token loss, log (dashed = base)", "Mean token loss (nats)")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"{protocol.get('protocol', 'v1')} rule, shuffled order, {len(seeds)} seeds — verdict: {protocol['verdict']}",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print("Wrote", output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    plot(Path(args.runs), analysis, protocol, analysis.get("metric", "cosine_separation"), Path(args.output))


if __name__ == "__main__":
    main()
