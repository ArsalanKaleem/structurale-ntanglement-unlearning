"""End-to-end v2 pipeline on a tiny local GPT-2: fine-tune, calibrate, run, analyze."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "fictional_authors_v1.jsonl"
CALIBRATION = ROOT / "data" / "fictional_authors_calibration_v1.jsonl"
WORDS = ["the", "fictional", "of", "is", "a", "for", "in", "was", "born", "island", "novel", "."]


def run_script(name, *args):
    old_argv = sys.argv
    try:
        sys.argv = [name, *map(str, args)]
        runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")
    finally:
        sys.argv = old_argv


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    import torch

    torch.manual_seed(0)
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocabulary), n_positions=64, n_embd=16, n_layer=2, n_head=2,
    ))
    path = tmp_path_factory.mktemp("tiny_model")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def make_config(tmp, tiny_model, method, seed, output_dir, dataset=MAIN, steps=3):
    return {
        "experiment": {
            "name": "pipeline_test", "seed": seed, "output_dir": str(output_dir),
            "checkpoint_retention": "resume_only", "shared_base_dir": str(tmp / "finetuned"),
        },
        "model": {"name_or_path": str(tiny_model), "revision": "main", "device": "cpu", "max_length": 32},
        "data": {
            "input_jsonl": str(dataset), "version": "test",
            **{f"max_{s}_examples": 8 for s in (
                "forget", "forget_update", "retain", "control", "control_update", "retain_update")},
        },
        "finetune": {
            "datasets": [str(MAIN), str(CALIBRATION)], "learning_rate": 0.01, "epochs": 3,
            "batch_size": 64, "seed": 0, "max_loss_ratio": 100.0,
        },
        "unlearning": {
            "method": method, "learning_rate": 0.001, "max_steps_per_request": 1,
            "batch_size": 1, "sequence_length": steps, "gradient_accumulation_steps": 1,
            "npo_beta": 0.1,
        },
        "evaluation": {
            "primary_structure_metric": "cosine_separation_forget_retain",
            "prompt_template": "{text}", "fixed_layer": -1, "extra_layers": [1],
            "max_eval_examples_per_split": 8,
        },
    }


def write_config(path, config):
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_v2_pipeline_end_to_end(tmp_path, tiny_model):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    first = write_config(tmp_path / "ft.yaml", make_config(tmp_path, tiny_model, "npo", 1, tmp_path / "x"))
    run_script("01b_finetune.py", "--config", first)
    report_path = tmp_path / "finetuned_report.json"
    report = json.loads(report_path.read_text())
    assert report["knowledge_gate_passed"]
    assert set(report["loss_before"]) == {MAIN.name, CALIBRATION.name}
    run_script("01b_finetune.py", "--config", first)  # idempotent second call

    calibration_root = tmp_path / "calibration"
    for lr in (0.001, 0.003):
        cfg = make_config(
            tmp_path, tiny_model, "npo", 1, calibration_root / "npo" / f"lr{lr}", dataset=CALIBRATION
        )
        cfg["unlearning"]["learning_rate"] = lr
        path = write_config(tmp_path / f"cal_{lr}.yaml", cfg)
        run_script("03_run_sequential.py", "--config", path, "--condition", "forget", "--order", "original")
    run_script(
        "09_select_hyperparameters.py", "--runs", calibration_root, "--finetune-report", report_path,
        "--dataset-name", CALIBRATION.name, "--output", tmp_path / "selected.json",
    )
    selection = json.loads((tmp_path / "selected.json").read_text())
    assert len(selection["by_method"]["npo"]["candidates"]) == 2

    main_root = tmp_path / "main" / "npo"
    for seed in (1, 2, 3):
        path = write_config(
            tmp_path / f"npo_{seed}.yaml",
            make_config(tmp_path, tiny_model, "npo", seed, main_root / f"seed_{seed}"),
        )
        run_script("02_run_baseline.py", "--config", path)
        for condition in ("forget", "control", "retain"):
            run_script("03_run_sequential.py", "--config", path, "--condition", condition, "--order", "shuffled")

    run_dir = main_root / "seed_1" / "condition_forget_shuffled"
    assert not list(run_dir.glob("checkpoint_step_*"))
    assert (run_dir / "reference_logprobs.json").is_file()
    assert (run_dir / "per_example" / "step_3.json").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert {"forget_update_mean_token_loss", "cosine_separation_forget_retain_L1"} <= set(metrics[0])
    assert not (main_root / "seed_1" / "base_model").exists()

    analysis_dir = tmp_path / "analysis"
    run_script(
        "06_analyze.py", "--input", main_root, "--output", analysis_dir,
        "--metric", "cosine_separation_forget_retain", "--bootstrap-samples", "50",
    )
    run_script(
        "07_apply_protocol.py", "--runs", main_root,
        "--analysis", analysis_dir / "analysis_cosine_separation_forget_retain.json",
        "--output", analysis_dir / "protocol.json", "--protocol", "v2",
        "--finetune-report", report_path, "--dataset-name", MAIN.name,
        "--primary-metric", "cosine_separation_forget_retain", "--horizon", "3",
    )
    verdict = json.loads((analysis_dir / "protocol.json").read_text())
    assert verdict["protocol"] == "v2" and verdict["confirmatory"]
    assert "valid_steps" in verdict["guard"]
    run_script(
        "10_spillover.py", "--input", main_root, "--dataset", MAIN,
        "--output", analysis_dir / "spillover.json", "--h2b-step", "2", "--bootstrap-samples", "50",
    )
    spill = json.loads((analysis_dir / "spillover.json").read_text())
    assert spill["n_seeds"] == 3
    assert len(spill["H2a_same_domain_spillover"]["per_seed"]) == 3

    ga_path = write_config(
        tmp_path / "ga.yaml", make_config(tmp_path, tiny_model, "ga", 1, tmp_path / "main" / "ga" / "seed_1")
    )
    run_script("02_run_baseline.py", "--config", ga_path)
    run_script("03_run_sequential.py", "--config", ga_path, "--condition", "forget", "--order", "shuffled")
    ga_meta = json.loads((tmp_path / "main" / "ga" / "seed_1" / "condition_forget_shuffled" / "run.json").read_text())
    assert ga_meta["status"] == "complete" and ga_meta["update_count"] == 3
