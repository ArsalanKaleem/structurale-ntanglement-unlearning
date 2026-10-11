import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    args = p.parse_args()
    from se_unlearning.utils.config import load_config
    cfg = load_config(args.config)
    out = Path(cfg["experiment"]["output_dir"])
    paths = list(out.glob("condition_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No condition metrics.csv found under {out}. Run scripts 02 and 03 first."
        )
    for path in paths:
        df = pd.read_csv(path)
        print(f"{path}: {len(df)} rows")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
