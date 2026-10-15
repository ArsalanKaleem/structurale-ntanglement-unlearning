import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "select_hp", Path(__file__).parents[1] / "scripts" / "09_select_hyperparameters.py"
)
select_hp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select_hp)

PRE = {"retain": 6.0, "control": 6.0, "forget_update": 6.0}


def write_run(root, name, lr, steps, retain_losses, control_losses, target_start, target_end):
    run_dir = root / name / "condition_forget_original"
    (run_dir / "per_example").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "status": "complete",
        "hyperparameters": {"method": "npo", "learning_rate": lr, "max_steps_per_request": steps},
        "update_examples": [{"id": "u1"}, {"id": "u2"}],
    }))
    rows = [
        {"step": t, "retain_mean_token_loss": r, "control_mean_token_loss": c}
        for t, (r, c) in enumerate(zip(retain_losses, control_losses))
    ]
    (run_dir / "metrics.json").write_text(json.dumps(rows))
    final = len(rows) - 1
    for step, value in ((0, target_start), (final, target_end)):
        (run_dir / "per_example" / f"step_{step}.json").write_text(
            json.dumps({"forget_update": {"u1": value, "u2": value, "u3": 0.0}})
        )
    return run_dir


def test_knowledge_fraction_scale():
    assert select_hp.knowledge_fraction(6.0, 1.0, 1.0) == 1.0
    assert select_hp.knowledge_fraction(6.0, 1.0, 6.0) == 0.0
    with pytest.raises(ValueError, match="not learned"):
        select_hp.knowledge_fraction(1.0, 1.0, 2.0)


def test_selects_most_effective_guard_passing_configuration(tmp_path):
    gentle = write_run(tmp_path, "gentle", 1e-6, 2, [1, 1.2, 1.3], [1, 1.1, 1.2], 1.0, 3.0)
    strong = write_run(tmp_path, "strong", 3e-6, 2, [1, 1.5, 1.9], [1, 1.2, 1.4], 1.0, 4.5)
    collapse = write_run(tmp_path, "collapse", 1e-5, 2, [1, 5, 9], [1, 5, 9], 1.0, 9.0)
    candidates = [select_hp.evaluate_run(d, PRE) for d in (gentle, strong, collapse)]

    by_name = {Path(c["run"]).parent.name: c for c in candidates}
    assert by_name["gentle"]["efficacy"] == pytest.approx(0.4)
    assert by_name["strong"]["passes_guard"] and by_name["strong"]["passes_efficacy"]
    assert not by_name["collapse"]["passes_guard"]
    result = select_hp.select(candidates)
    assert result["status"] == "selected"
    assert result["selected"]["learning_rate"] == 3e-6


def test_reports_when_no_configuration_is_admissible(tmp_path):
    weak = write_run(tmp_path, "weak", 1e-6, 2, [1, 1, 1], [1, 1, 1], 1.0, 1.5)
    collapse = write_run(tmp_path, "collapse", 1e-5, 2, [1, 5, 9], [1, 5, 9], 1.0, 9.0)
    assert select_hp.select([select_hp.evaluate_run(weak, PRE)])["status"] == "efficacy_not_met"
    assert select_hp.select([select_hp.evaluate_run(collapse, PRE)]) == {
        "status": "no_config_passes_guard", "selected": None,
    }
