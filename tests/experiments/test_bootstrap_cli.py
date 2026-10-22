"""Manuscript tables: apparent estimates carry no interval, corrected ones do."""

import numpy as np
import pandas as pd
import pytest

from experiments.bootstrap_cli import (
    ci_cell,
    format_metrics_report,
    format_subgroup_report,
    point_cell,
)
from experiments.subgroup_cli import summarise_subgroups


def test_point_cell_rounds_like_ci_cell():
    assert point_cell(0.065) == "0.07"     # half-up on the written value
    assert point_cell(-0.001) == "0.00"    # no "-0.00"
    assert ci_cell(0.065, -0.001, 0.125) == "0.07 (0.00 to 0.13)"


def test_metrics_report_apparent_has_no_interval():
    metrics_df = pd.DataFrame([{
        "metric": "auc",
        "apparent": 0.9227,
        "corrected": 0.9083,
        "corrected_ci_lower": 0.8877,
        "corrected_ci_upper": 0.9274,
    }])
    report = format_metrics_report(metrics_df, decimals=3)
    assert list(report.columns) == ["Metric", "Apparent", "Optimism-corrected (95% CI)"]
    assert report.loc[0, "Apparent"] == "0.923"
    assert report.loc[0, "Optimism-corrected (95% CI)"] == "0.908 (0.888 to 0.927)"


def test_subgroup_tables_drop_apparent_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0, 1)
    masks = {"Overall": np.ones(200, dtype=bool)}
    optimism_df = pd.DataFrame({"Overall": [0.01, 0.02, 0.03]})
    apparent_df = pd.DataFrame({"Overall": [0.70, 0.75, np.nan, 0.80]})

    summary = summarise_subgroups(y, p, masks, optimism_df, apparent_df)
    assert not any("apparent_ci" in c for c in summary.columns)
    r = summary.iloc[0]
    assert r.auc_corrected == pytest.approx(r.auc_apparent - 0.02)
    # Location-shifted: percentiles of the per-resample apparent AUC minus the optimism.
    lo, hi = np.nanpercentile([0.70, 0.75, 0.80], [2.5, 97.5])
    assert r.auc_corrected_ci_lower == pytest.approx(lo - 0.02)
    assert r.auc_corrected_ci_upper == pytest.approx(hi - 0.02)

    report = format_subgroup_report(summary)
    assert "Apparent AUC" in report.columns
    assert "(" not in report.loc[0, "Apparent AUC"]
    assert "(" in report.loc[0, "Optimism-corrected AUC (95% CI)"]
