import logging
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from datetime import datetime, timezone
import torch
from se_unlearning.utils.artifacts import write_json_atomic

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

def write_run_metadata(output_dir: str | Path, config: dict, run_info: dict | None = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": (run_info or {}).get("device"),
        "package_versions": _package_versions(),
        "config": config,
        **(run_info or {}),
    }
    write_json_atomic(out / "run.json", meta)


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "tokenizers",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "tqdm",
        "pytest",
        "structural-entanglement-unlearning",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
