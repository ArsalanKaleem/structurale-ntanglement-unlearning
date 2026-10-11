import json
import logging
import platform
from pathlib import Path
from datetime import datetime, timezone
import torch

def setup_logger(output_dir: str | Path) -> logging.Logger:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("se_unlearning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out / "run.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def write_run_metadata(output_dir: str | Path, config: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": config,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
