import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    if "step" not in df:
        raise ValueError("Input CSV must contain a 'step' column.")
    metric_cols = [
        c for c in ["forget_mean_token_loss", "retain_mean_token_loss",
                    "control_mean_token_loss", "cosine_separation"] if c in df
    ]
    if not metric_cols:
        raise ValueError("No recognized metric columns in CSV.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in metric_cols:
        ax.plot(df["step"], df[col], marker="o", label=col)
    ax.set_xlabel("Sequential unlearning step")
    ax.set_ylabel("Metric value")
    ax.set_title("Sequential unlearning pilot metrics")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
