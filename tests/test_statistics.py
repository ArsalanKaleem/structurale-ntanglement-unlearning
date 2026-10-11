import pytest

from se_unlearning.metrics.statistics import paired_effect_summary


def test_paired_effect_summary_reports_effect_and_deterministic_interval():
    first = paired_effect_summary([1.0, 2.0, 3.0], seed=7, bootstrap_samples=1000)
    second = paired_effect_summary([1.0, 2.0, 3.0], seed=7, bootstrap_samples=1000)

    assert first["n_seeds"] == 3
    assert first["mean_paired_effect"] == pytest.approx(2.0)
    assert first["standardized_paired_effect_dz"] == pytest.approx(2.0)
    assert first["bootstrap_95_percentile_interval"] == second[
        "bootstrap_95_percentile_interval"
    ]


def test_single_seed_does_not_claim_uncertainty_or_standardized_effect():
    summary = paired_effect_summary([0.5], seed=4)

    assert summary["mean_paired_effect"] == 0.5
    assert summary["bootstrap_95_percentile_interval"] is None
    assert summary["standardized_paired_effect_dz"] is None
