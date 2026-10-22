"""Tests for exact SHAP values on the linear mortality model.

Uses the same tiny synthetic dataset shape as tests/validation/test_bootstrap.py.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from explainability.shap_values import (
    compute_shap,
    display_frame,
    importance_table,
    onehot_groups,
    spline_groups,
)
from src.validation.bootstrap import develop_model
from tests.validation.test_bootstrap import _PIPELINE_CFG, _X, _Y


@pytest.fixture(scope="module")
def fitted():
    """Fit the real develop_model() once on the synthetic dataset."""
    return develop_model(_X, _Y, _PIPELINE_CFG)


@pytest.fixture(scope="module")
def fitted_spline():
    """Same model with age splined, so there are basis columns to group."""
    cfg = copy.deepcopy(_PIPELINE_CFG)
    cfg["features"]["spline"] = {
        "LEEFTIJD_EXACT": {"knot_quantiles": [0.10, 0.50, 0.90]}
    }
    return develop_model(_X, _Y, cfg)


def test_shap_values_are_additive(fitted):
    """base_value + sum(phi_i) must equal the model's decision function."""
    expl = compute_shap(fitted, _X)
    X_pp = fitted.preprocessor.transform(_X)
    expected = fitted.model.decision_function(X_pp)
    np.testing.assert_allclose(
        expl.base_values + expl.values.sum(axis=1), expected, atol=1e-8
    )


def test_shap_matches_closed_form(fitted):
    """For a linear model, phi_ij == beta_j * (x_ij - mean(x_j))."""
    expl = compute_shap(fitted, _X)
    X_pp = fitted.preprocessor.transform(_X).to_numpy(dtype=float)
    beta = fitted.model.coef_[0]
    np.testing.assert_allclose(
        expl.values, beta * (X_pp - X_pp.mean(axis=0)), atol=1e-8
    )


def test_mean_signed_shap_is_zero(fitted):
    """Using the full cohort as background centres every feature's SHAP at zero."""
    expl = compute_shap(fitted, _X)
    np.testing.assert_allclose(expl.values.mean(axis=0), 0.0, atol=1e-8)


def test_display_frame_keeps_original_units(fitted):
    """Display data must be unscaled but share the model's column layout."""
    X_pp = fitted.preprocessor.transform(_X)
    disp = display_frame(fitted.preprocessor, _X)
    assert list(disp.columns) == list(X_pp.columns)
    # Age is Yeo-Johnson standardized in the model input but not in the display frame.
    assert disp["LEEFTIJD_EXACT"].mean() == pytest.approx(_X["LEEFTIJD_EXACT"].mean(), rel=0.05)
    assert abs(X_pp["LEEFTIJD_EXACT"].mean()) < 1e-6


def test_display_names_applied(fitted):
    """Configured labels replace encoded column names, unmapped ones pass through."""
    expl = compute_shap(fitted, _X, display_names={"LEEFTIJD_EXACT": "Age (years)"})
    assert "Age (years)" in expl.feature_names
    assert "BMI" in expl.feature_names


def test_spline_groups_maps_basis_to_source(fitted_spline):
    """Every basis column of a splined predictor maps back to that predictor."""
    assert spline_groups(fitted_spline) == {
        "LEEFTIJD_EXACT_s1": "LEEFTIJD_EXACT",
        "LEEFTIJD_EXACT_s2": "LEEFTIJD_EXACT",
    }


def test_grouping_sums_the_basis_columns(fitted_spline):
    """The grouped age row must equal the sum of its basis columns, exactly."""
    groups = spline_groups(fitted_spline)
    ungrouped = compute_shap(fitted_spline, _X)
    grouped = compute_shap(fitted_spline, _X, groups=groups)

    cols = list(ungrouped.feature_names)
    basis = ungrouped.values[:, [cols.index(c) for c in groups]]
    age = grouped.values[:, list(grouped.feature_names).index("LEEFTIJD_EXACT")]

    assert grouped.values.shape[1] == ungrouped.values.shape[1] - 1
    np.testing.assert_allclose(age, basis.sum(axis=1), atol=1e-12)


def test_grouping_preserves_additivity(fitted_spline):
    """Summing columns cannot change the prediction the values decompose."""
    grouped = compute_shap(fitted_spline, _X, groups=spline_groups(fitted_spline))
    X_pp = fitted_spline.preprocessor.transform(_X)
    np.testing.assert_allclose(
        grouped.base_values + grouped.values.sum(axis=1),
        fitted_spline.model.decision_function(X_pp), atol=1e-8,
    )


def test_grouped_row_displays_the_original_value(fitted_spline):
    """The age row must show age in years, not a basis coordinate."""
    grouped = compute_shap(fitted_spline, _X, groups=spline_groups(fitted_spline))
    shown = grouped.data[:, list(grouped.feature_names).index("LEEFTIJD_EXACT")]
    np.testing.assert_allclose(shown, _X["LEEFTIJD_EXACT"].to_numpy(dtype=float))


def test_onehot_groups_maps_dummies_to_source(fitted):
    """Every INTENTIE dummy maps back to the categorical predictor it expands."""
    assert onehot_groups(fitted) == {
        "INTENTIE_fall_high": "INTENTIE_cat",
        "INTENTIE_MVA": "INTENTIE_cat",
        "INTENTIE_other": "INTENTIE_cat",
        "INTENTIE_violence": "INTENTIE_cat",
    }


def test_onehot_grouping_sums_the_dummies(fitted):
    """The mechanism row must equal the sum of its dummy columns, exactly."""
    groups = onehot_groups(fitted)
    ungrouped = compute_shap(fitted, _X)
    grouped = compute_shap(fitted, _X, groups=groups)

    cols = list(ungrouped.feature_names)
    dummies = ungrouped.values[:, [cols.index(c) for c in groups]]
    mechanism = grouped.values[:, list(grouped.feature_names).index("INTENTIE_cat")]

    assert grouped.values.shape[1] == ungrouped.values.shape[1] - (len(groups) - 1)
    np.testing.assert_allclose(mechanism, dummies.sum(axis=1), atol=1e-12)


def test_onehot_grouped_row_shows_the_level(fitted):
    """The collapsed row shows each patient's mechanism, reference level included."""
    grouped = compute_shap(fitted, _X, groups=onehot_groups(fitted))
    shown = grouped.data[:, list(grouped.feature_names).index("INTENTIE_cat")]
    assert list(shown) == list(_X["INTENTIE_cat"])
    assert "fall_low" in set(shown)  # the reference level has no dummy of its own


def test_importance_table_sorted(fitted):
    """Rows come back ranked by mean |SHAP|, descending."""
    df = importance_table(compute_shap(fitted, _X))
    assert isinstance(df, pd.DataFrame)
    assert df["mean_abs_shap"].is_monotonic_decreasing
    assert (df["min_shap"] <= df["max_shap"]).all()
