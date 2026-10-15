import hashlib
import json
from pathlib import Path
import re
import shutil


def _directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "base_manifest.json":
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_base_checkpoint(
    model, tokenizer, checkpoint_dir: str | Path, model_name: str, revision: str
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Base checkpoint already exists: {checkpoint_dir}")
    temporary_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    manifest = {
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None) or revision,
        "checkpoint_sha256": _directory_sha256(temporary_dir),
    }
    (temporary_dir / "base_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    temporary_dir.replace(checkpoint_dir)
    return manifest


def read_base_manifest(checkpoint_dir: str | Path) -> dict:
    path = Path(checkpoint_dir) / "base_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Base checkpoint manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "requested_revision",
        "resolved_revision",
        "checkpoint_sha256",
    ):
        if not manifest.get(key):
            raise ValueError(f"Base checkpoint manifest has no {key!r}: {path}")
    return manifest


def validate_base_checkpoint(
    checkpoint_dir: str | Path, model_name: str, revision: str
) -> dict:
    manifest = read_base_manifest(checkpoint_dir)
    if manifest["model_name"] != model_name or manifest["requested_revision"] != revision:
        raise ValueError(
            "Base checkpoint does not match the configured model and requested revision"
        )
    checkpoint_hash = _directory_sha256(Path(checkpoint_dir))
    if manifest["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError(
            f"Base checkpoint contents do not match their manifest: {checkpoint_dir}"
        )
    return manifest


def validate_model_revision(model_name: str, revision: str) -> None:
    if Path(model_name).exists():
        return
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(
            "Remote model revision must be pinned to a full 40-character commit SHA "
            "before downloading or running the pilot"
        )
