import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    args = p.parse_args()
    # The sequential runner already evaluates every checkpoint and writes the CSV.
    from se_unlearning.utils.config import load_config
    cfg = load_config(args.config)
    out = Path(cfg["experiment"]["output_dir"])
    paths = list(out.glob("sequential_*/sequential_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No sequential_metrics.csv found under {out}. Run script 03 first.")
    for path in paths:
        df = pd.read_csv(path)
        summary_path = path.with_name("summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"{path}: {len(df)} rows -> {summary_path}")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
