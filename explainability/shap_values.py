"""Exact SHAP values for the fitted LogisticRegression mortality model.

The final model is linear, so interventional SHAP values are exact and closed
form: phi_ij = beta_j * (x_ij - mean(x_j)). They are expressed on the log-odds
scale, where they are additive: base_value + sum_j(phi_ij) equals the model's
decision function for patient i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.preprocessing.transformers import spline_columns


def display_frame(preprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """Re-run preprocessing without the value transforms, keeping original units.

    The Yeo-Johnson step and the scaler only rescale columns in place and never
    add, drop, or reorder them, so both can be skipped to get a frame with values
    a clinician can read (age in years, ISS in points) instead of z-scores.

    A splined predictor does change the layout, expanding into several basis
    columns. Transformer.expand_display() reproduces that layout while repeating
    the predictor's original value across its basis columns, so those columns
    display in years rather than in basis coordinates.

    Args:
        preprocessor: Fitted preprocessing Pipeline (cleaner -> imputer ->
            transformer -> encoder).
        X: Raw feature DataFrame.

    Returns:
        DataFrame with the model's post-encoding columns in original units.
    """
    steps = preprocessor.named_steps
    transformer = steps["transformer"]
    transformer = getattr(transformer, "transformer", transformer)

    df = steps["cleaner"].transform(X)
    df = steps["imputer"].transform(df)
    df = transformer.expand_display(df)
    return steps["encoder"].transform(df)


def spline_groups(fitted_model) -> dict[str, str]:
    """Map each spline basis column to the predictor it expands.

    Args:
        fitted_model: _FittedModel whose spline_knots give the fitted basis sizes.

    Returns:
        Mapping of basis column name -> source predictor name.
    """
    return {
        name: col
        for col, knots in fitted_model.spline_knots.items()
        for name in spline_columns(col, len(knots) - 1)
    }


def _categorical(fitted_model) -> tuple[str, str, str]:
    """Return (source column, dummy column prefix, reference level) of the encoder."""
    cfg = fitted_model.preprocessor.named_steps["encoder"].config
    source = cfg["features"]["categorical"][0]
    return source, source.replace("_cat", "_"), cfg["encoding"]["intentie_reference"]


def onehot_groups(fitted_model) -> dict[str, str]:
    """Map each one-hot dummy column to the categorical predictor it expands.

    Args:
        fitted_model: _FittedModel whose preprocessor holds the fitted encoder.

    Returns:
        Mapping of dummy column name -> source predictor name.
    """
    source, prefix, _ = _categorical(fitted_model)
    return {c: source for c in fitted_model.feature_names if c.startswith(prefix)}


def _onehot_levels(fitted_model, X_display: pd.DataFrame) -> dict[str, pd.Series]:
    """Read each patient's category back out of the one-hot dummy columns.

    The encoder drops the reference level, so a patient with every dummy at zero
    is a reference-level patient and takes that level's name.

    Args:
        fitted_model: _FittedModel whose preprocessor holds the fitted encoder.
        X_display: Original-unit frame in model column order.

    Returns:
        Mapping of source predictor name -> per-patient level name.
    """
    source, prefix, reference = _categorical(fitted_model)
    dummies = [c for c in X_display.columns if c.startswith(prefix)]
    if not dummies:
        return {}
    # The reference level is appended as an always-true last column, so argmax
    # falls through to it when none of the dummies is set.
    levels = np.array([c[len(prefix):] for c in dummies] + [reference])
    active = np.column_stack([X_display[dummies].to_numpy() == 1,
                              np.ones(len(X_display), dtype=bool)])
    return {source: pd.Series(levels[active.argmax(axis=1)], index=X_display.index)}


def _collapse(values: np.ndarray, X_display: pd.DataFrame, groups: dict[str, str],
              levels: dict[str, pd.Series] | None = None):
    """Sum the SHAP columns of each grouped predictor into a single column.

    Args:
        values: (n_patients, n_columns) SHAP values in model column order.
        X_display: Original-unit frame with the same columns.
        groups: Mapping of column name -> group name, from spline_groups() or
            onehot_groups().
        levels: Optional per-patient level names, from _onehot_levels(). A group
            listed here shows its level instead of a dummy indicator.

    Returns:
        (summed values, display frame) with one column per group. A group takes
        the position and the display value of its first member, which is the
        predictor's original value: Transformer.expand_display() repeats that
        value across the basis columns.
    """
    keys = [groups.get(c, c) for c in X_display.columns]
    members = {key: [i for i, k in enumerate(keys) if k == key]
               for key in dict.fromkeys(keys)}
    summed = np.column_stack([values[:, idx].sum(axis=1) for idx in members.values()])
    levels = levels or {}
    display = pd.DataFrame({
        key: (levels[key].to_numpy() if key in levels
              else X_display.iloc[:, idx[0]].to_numpy())
        for key, idx in members.items()
    })
    return summed, display


def compute_shap(
    fitted_model,
    X: pd.DataFrame,
    display_names: dict[str, str] | None = None,
    groups: dict[str, str] | None = None,
) -> shap.Explanation:
    """Compute exact log-odds SHAP values for every patient in X.

    Args:
        fitted_model: _FittedModel from bootstrap_validate() / final_model.joblib.
        X: Raw feature DataFrame (same schema as training input).
        display_names: Optional mapping of encoded column name -> plot label.
        groups: Optional mapping of column name -> group name; grouped columns are
            summed into one. Intended for a splined predictor, whose basis columns
            only mean anything together - individually their values partially
            cancel, so a waterfall listing them separately shows rows that are not
            that predictor's effect - and for the one-hot dummies of the
            categorical predictor, which are mutually exclusive: at most one is
            set per patient, so their separate rows are one effect split across
            levels the patient does not have. Summing is exact, since the values
            are additive on the log-odds scale.

    Returns:
        shap.Explanation with values on the log-odds scale, .data holding the
        original-unit feature values, and .feature_names holding plot labels.
        A collapsed categorical column holds its level name, which makes .data
        object-typed rather than float.

    Raises:
        ValueError: If the additivity check against decision_function fails.
    """
    X_model = fitted_model.preprocessor.transform(X)
    X_display = display_frame(fitted_model.preprocessor, X)
    if list(X_display.columns) != list(X_model.columns):
        raise ValueError(
            "Display frame columns differ from model input columns; the "
            "preprocessing pipeline no longer keeps the scaler layout-neutral."
        )

    # max_samples defaults to 100; the full cohort is used as background so the
    # baseline is the true cohort mean and the SHAP values stay exact.
    masker = shap.maskers.Independent(X_model, max_samples=len(X_model))
    explainer = shap.LinearExplainer(fitted_model.model, masker)
    values = np.asarray(explainer.shap_values(X_model), dtype=float)
    base = float(np.ravel(explainer.expected_value)[0])

    expected = fitted_model.model.decision_function(X_model)
    if not np.allclose(values.sum(axis=1) + base, expected, atol=1e-6):
        raise ValueError("SHAP values are not additive w.r.t. decision_function.")

    # Grouped after the additivity check, which summing preserves.
    if groups:
        values, X_display = _collapse(values, X_display, groups,
                                      _onehot_levels(fitted_model, X_display))

    names = list(X_display.columns)
    if display_names:
        names = [display_names.get(c, c) for c in names]

    return shap.Explanation(
        values=values,
        base_values=np.full(len(X_model), base),
        data=X_display.to_numpy(),
        feature_names=names,
    )


def importance_table(expl: shap.Explanation) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Args:
        expl: Explanation from compute_shap().

    Returns:
        DataFrame with columns feature, mean_abs_shap, min_shap, max_shap,
        sorted descending by mean_abs_shap. The signed min/max show in which
        direction and how far the feature can push an individual prediction;
        the mean signed SHAP is zero by construction and is not reported.
    """
    df = pd.DataFrame({
        "feature": expl.feature_names,
        "mean_abs_shap": np.abs(expl.values).mean(axis=0),
        "min_shap": expl.values.min(axis=0),
        "max_shap": expl.values.max(axis=0),
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
