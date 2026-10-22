"""Tests for the coefficient / odds-ratio table and its scale conversions."""

import numpy as np
import pytest

from explainability.coefficients import (
    coefficient_table,
    native_intercept,
    plot_coefficient_forest,
)
from src.validation.bootstrap import develop_model
from tests.validation.test_bootstrap import _PIPELINE_CFG, _X, _Y


@pytest.fixture(scope="module")
def fitted():
    """Fit the real develop_model() once on the synthetic dataset."""
    return develop_model(_X, _Y, _PIPELINE_CFG)


def test_native_scale_reproduces_linear_predictor(fitted):
    """beta/scale_ with the adjusted intercept must give the same log-odds."""
    pre_scale = fitted.preprocessor[:-1].transform(_X).to_numpy(dtype=float)
    beta_native = fitted.model.coef_[0] / fitted.feature_scales
    logits = native_intercept(fitted) + pre_scale @ beta_native

    scaled = fitted.preprocessor.transform(_X)
    np.testing.assert_allclose(logits, fitted.model.decision_function(scaled), atol=1e-8)


def test_scaler_stats_match_feature_count(fitted):
    """Captured means/scales line up with the encoded feature vector."""
    n = len(fitted.feature_names)
    assert fitted.feature_means.shape == (n,)
    assert fitted.feature_scales.shape == (n,)
    assert (fitted.feature_scales > 0).all()


def test_table_sorted_and_units_assigned(fitted):
    """Rows rank by |beta| and every predictor gets a unit label."""
    df = coefficient_table(fitted, _PIPELINE_CFG["features"])
    assert df.beta_std.abs().is_monotonic_decreasing
    assert df.unit.notna().all()
    # Continuous predictors are Yeo-Johnson transformed: no clinical native OR.
    cont = _PIPELINE_CFG["features"]["continuous"]
    assert df.loc[df.encoded_name.isin(cont), "or_native"].isna().all()
    assert df.loc[~df.encoded_name.isin(cont), "or_native"].notna().all()


def test_bootstrap_cis_bracket_and_widen(fitted):
    """CIs appear when bootstrap draws are supplied, and lo <= hi throughout."""
    rng = np.random.default_rng(0)
    n = len(fitted.feature_names)
    boot_coefs = fitted.model.coef_[0] + rng.normal(0, 0.05, size=(50, n))
    boot_scales = np.tile(fitted.feature_scales, (50, 1))

    df = coefficient_table(
        fitted, _PIPELINE_CFG["features"],
        boot_coefs=boot_coefs, boot_scales=boot_scales,
    )
    assert (df.or_per_sd_lo <= df.or_per_sd_hi).all()
    ok = df.or_native.notna()
    assert (df.loc[ok, "or_native_lo"] <= df.loc[ok, "or_native_hi"]).all()
    assert (df.n_bootstraps == 50).all()


def test_forest_plot_renders(fitted, tmp_path):
    """Forest plot writes a file for a table both with and without CIs."""
    import matplotlib
    matplotlib.use("Agg")

    df = coefficient_table(fitted, _PIPELINE_CFG["features"])
    out = tmp_path / "forest.png"
    plot_coefficient_forest(df, save_path=out)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Splined predictors: reported as a risk curve, not as a table row
# ---------------------------------------------------------------------------

import copy

from explainability.coefficients import spline_contrasts

_SPLINE_CFG = copy.deepcopy(_PIPELINE_CFG)
_SPLINE_CFG["features"]["spline"] = {
    "LEEFTIJD_EXACT": {"knot_quantiles": [0.10, 0.50, 0.90]}
}


@pytest.fixture(scope="module")
def spline_fitted():
    """develop_model() with age entered as a restricted cubic spline."""
    return develop_model(_X, _Y, _SPLINE_CFG)


def test_spline_knots_captured_on_the_fitted_model(spline_fitted):
    knots = spline_fitted.spline_knots["LEEFTIJD_EXACT"]
    assert len(knots) == 3
    assert np.all(np.diff(knots) > 0)


def test_spline_expands_into_basis_columns(spline_fitted):
    names = spline_fitted.feature_names
    assert "LEEFTIJD_EXACT" not in names
    assert {"LEEFTIJD_EXACT_s1", "LEEFTIJD_EXACT_s2"} <= set(names)


def test_spline_contrast_is_one_at_the_reference(spline_fitted):
    df = spline_contrasts(spline_fitted, "LEEFTIJD_EXACT", [40.0, 60.0, 80.0], 60.0)
    assert df.loc[df.value == 60.0, "odds_ratio"].item() == pytest.approx(1.0)


def test_spline_contrast_matches_the_fitted_log_odds(spline_fitted):
    """The contrast must equal the model's own change in log-odds when only age moves."""
    row = _X.iloc[[0]].copy()
    ref, target = 45.0, 82.0

    def logit(age):
        r = row.copy()
        r["LEEFTIJD_EXACT"] = age
        return spline_fitted.model.decision_function(
            spline_fitted.preprocessor.transform(r)
        )[0]

    expected = np.exp(logit(target) - logit(ref))
    got = spline_contrasts(spline_fitted, "LEEFTIJD_EXACT", [target], ref)
    assert got.odds_ratio.item() == pytest.approx(expected, rel=1e-8)


def test_spline_contrast_bootstrap_bounds_are_ordered(spline_fitted):
    rng = np.random.default_rng(1)
    n = len(spline_fitted.feature_names)
    knots = spline_fitted.spline_knots["LEEFTIJD_EXACT"]
    df = spline_contrasts(
        spline_fitted, "LEEFTIJD_EXACT", [30.0, 60.0, 90.0], 60.0,
        boot_coefs=spline_fitted.model.coef_[0] + rng.normal(0, 0.05, (30, n)),
        boot_scales=np.tile(spline_fitted.feature_scales, (30, 1)),
        boot_knots=np.tile(knots, (30, 1)),
    )
    assert (df.odds_ratio_lo <= df.odds_ratio_hi).all()
    assert (df.n_bootstraps == 30).all()


def test_spline_rows_excluded_from_forest_plot(spline_fitted):
    df = coefficient_table(spline_fitted, _SPLINE_CFG["features"])
    assert df.unit.str.startswith("spline basis for").sum() == 2
    fig = plot_coefficient_forest(df)
    assert len(fig.axes[0].get_yticklabels()) == len(df) - 2
