"""Waterfall pooling must preserve the prediction and apply the stated threshold."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from explainability.plots import (
    _probability_contributions,
    _waterfall_rows,
    _waterfall_xlim,
    plot_waterfall,
)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_threshold_is_a_share_of_the_total_and_pools_its_boundary(scale):
    # Shares of the summed |pp| are 50, 30, 5, 5, 4 and 6 percent at any scale,
    # so a low-risk patient with tiny contributions is split the same way.
    pp = np.array([50.0, -30.0, 5.0, -5.0, 4.0, -6.0]) * scale
    names = [f"predictor_{i}" for i in range(len(pp))]
    rows = _waterfall_rows(pp, names, np.zeros(len(pp)), top_n=1,
                           group_threshold_pct=5.0)

    # All >5% contributions survive even with top_n=1; the 5% boundaries pool.
    assert {name for name, _ in rows[:-1]} == {
        "0 = predictor_0", "0 = predictor_1", "0 = predictor_5",
    }
    assert rows[-1][0] == "Other predictors (n = 3)"
    assert rows[-1][1] == pytest.approx(4.0 * scale)  # signed sum, not magnitudes
    assert sum(value for _, value in rows) == pytest.approx(pp.sum())


@pytest.mark.parametrize("pp", [np.full(20, 0.5), np.tile([0.5, -0.5], 10)])
def test_all_small_contributions_form_one_bar_even_if_net_exceeds_threshold(pp):
    names = [f"predictor_{i}" for i in range(len(pp))]
    rows = _waterfall_rows(pp, names, np.zeros(len(pp)),
                           top_n=15, group_threshold_pct=5.0)
    assert len(rows) == 1
    assert rows[0][1] == pytest.approx(pp.sum())


def test_axis_does_not_clip_intermediate_contribution_sums():
    # From 5% through -3% to 12%; only the start and end are probabilities.
    lo, hi = _waterfall_xlim([("increase", 15.0), ("decrease", -8.0)], 0.05, 0.12)
    assert lo < -0.03
    assert hi > 0.12


def test_grouped_waterfall_still_ends_at_original_prediction(tmp_path):
    expl = SimpleNamespace(
        values=np.array([[1.2, -0.6, 0.04, -0.03, 0.01]]),
        base_values=np.array([-2.0]),
        feature_names=list("abcde"),
        data=np.zeros((1, 5)),
    )
    pp, p_base, p_pred = _probability_contributions(expl, 0)
    path = tmp_path / "waterfall.png"
    fig = plot_waterfall(expl, 0, group_threshold_pct=5.0, save_path=path)
    try:
        bars = fig.axes[0].patches
        assert len(bars) == 3
        assert f"Reference {p_base:.1%}" in [t.get_text() for t in fig.axes[0].texts]
        assert bars[0].get_xy()[0, 0] == pytest.approx(p_base)
        assert bars[-1].get_xy()[2, 0] == pytest.approx(p_pred)
        assert sum(bar.get_xy()[2, 0] - bar.get_xy()[0, 0] for bar in bars) == (
            pytest.approx(pp.sum() / 100)
        )
        assert path.is_file()
    finally:
        plt.close(fig)
