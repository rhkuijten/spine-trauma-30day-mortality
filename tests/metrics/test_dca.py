import numpy as np

from src.metrics.metrics import dca


def test_dca_default_uses_full_threshold_range():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.25, 0.75, 0.95])

    out = dca(y_true, y_prob)

    assert np.isclose(out["threshold"].iloc[0], 0.01)
    assert np.isclose(out["threshold"].iloc[-1], 0.99)
    assert len(out) == 99


def test_dca_high_threshold_treat_all_not_clipped():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.25, 0.75, 0.95])

    out = dca(y_true, y_prob)

    high = out.loc[np.isclose(out["threshold"], 0.99)].iloc[0]
    assert high["net_benefit_all"] < 0
    assert high["std_net_benefit_all"] < 0

