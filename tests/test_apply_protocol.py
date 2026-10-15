import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "apply_protocol", Path(__file__).parents[1] / "scripts" / "07_apply_protocol.py"
)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def summaries(reference, means, half_width=0.01, order="shuffled"):
    return [
        {
            "order": order, "step": step, "control_condition": reference, "n_seeds": 5,
            "mean_paired_effect": mean,
            "bootstrap_95_percentile_interval": [mean - half_width, mean + half_width],
        }
        for step, mean in enumerate(means, start=1)
    ]


def analysis(control, retain, no_update=None):
    flat = [0.0] * 16
    return {"summaries": [
        *summaries("control", control),
        *summaries("retain", retain),
        *summaries("no_update", no_update or flat),
    ]}


def test_tolerance_uses_ten_percent_of_baseline_with_floor():
    assert protocol.tolerance(-0.5) == pytest.approx(0.05)
    assert protocol.tolerance(0.005) == 0.001


def test_inversion_against_both_references_is_supported():
    up_then_down = [0.1] * 4 + [0.0] * 8 + [-0.1] * 4
    result = protocol.apply_rule(analysis(up_then_down, up_then_down), baseline=0.5)
    assert result["tau"] == pytest.approx(0.05)
    assert result["verdict"] == "supported"
    assert result["references"]["control"]["inversion_patterns"] == ["positive_then_negative"]


def test_control_only_inversion_is_partial():
    up_then_down = [0.1] * 4 + [0.0] * 8 + [-0.1] * 4
    result = protocol.apply_rule(analysis(up_then_down, [0.1] * 16), baseline=0.5)
    assert result["verdict"].startswith("partial")


def test_opposite_inversion_directions_do_not_count_as_support():
    up_then_down = [0.1] * 8 + [-0.1] * 8
    down_then_up = [-0.1] * 8 + [0.1] * 8
    result = protocol.apply_rule(analysis(up_then_down, down_then_up), baseline=0.5)
    assert result["verdict"].startswith("partial")


def test_below_tolerance_or_interval_touching_zero_is_not_reliable():
    small = [0.04] * 8 + [-0.04] * 8
    assert protocol.apply_rule(analysis(small, small), baseline=0.5)["verdict"] == "not detected"
    wide = analysis([0.1] * 8 + [-0.1] * 8, [0.1] * 8 + [-0.1] * 8)
    for summary in wide["summaries"]:
        summary["bootstrap_95_percentile_interval"] = [-1.0, 1.0]
    assert protocol.apply_rule(wide, baseline=0.5)["verdict"] == "not detected"


def test_monotone_effect_is_not_an_inversion():
    rising = [0.01 * step for step in range(1, 17)]
    result = protocol.apply_rule(analysis(rising, rising), baseline=0.05)
    assert result["references"]["control"]["reliable_positive_steps"]
    assert result["verdict"] == "not detected"


def test_missing_steps_are_rejected():
    bad = analysis([0.0] * 16, [0.0] * 16)
    bad["summaries"] = [s for s in bad["summaries"] if s["step"] != 7]
    with pytest.raises(ValueError, match="expected steps"):
        protocol.apply_rule(bad, baseline=0.5)
