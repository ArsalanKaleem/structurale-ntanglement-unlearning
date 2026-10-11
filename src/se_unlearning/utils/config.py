from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping.")
    required = {"experiment", "model", "data", "unlearning", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    data_version = config["data"].get("version")
    if not data_version or str(data_version).lower().startswith("replace"):
        raise ValueError("data.version must be a fixed, non-placeholder dataset version")
    metric = config["evaluation"].get("primary_structure_metric")
    if metric != "cosine_separation":
        raise ValueError(
            "The frozen primary_structure_metric must be 'cosine_separation'; "
            f"got {metric!r}"
        )
    prompt_template = config["evaluation"].get("prompt_template", "{text}")
    if not isinstance(prompt_template, str) or "{text}" not in prompt_template:
        raise ValueError("evaluation.prompt_template must contain the {text} field")
    return config
