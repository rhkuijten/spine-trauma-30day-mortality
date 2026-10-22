"""Coefficient / odds-ratio table and forest plot for the final model.

Two scales are reported for every predictor:

  standardized  The scale the model is actually fitted on (Scaler runs last, so
                every predictor including dummies is mean 0 / SD 1). Coefficients
                are directly comparable across predictors and this is the scale
                the ridge penalty acts on.
  native        beta / scale_, i.e. per one unit of the predictor as it enters
                the scaler. For binary and dummy predictors that is present vs
                absent, so exp() is a genuine odds ratio; for ordinal predictors
                it is per grade. The four continuous predictors reach the scaler
                already Yeo-Johnson transformed, so their native unit is not a
                clinical unit and no native OR is reported for them.

Splined predictors are the exception to both. Their basis columns only mean
anything together, so no row of this table is a reportable effect for them; they
are reported instead as a risk curve against a reference value by
spline_contrasts() / plot_spline_curve(), and are left out of the forest plot.

Coefficients are ridge-penalised, so analytic standard errors do not apply. All
intervals here are bootstrap percentile intervals over full model redevelopment
(BootstrapResult.boot_coefs / boot_scales / boot_spline_knots). Being percentiles
of penalised, and therefore deliberately biased, estimates, they describe how far
the estimate moves under resampling rather than carrying nominal 95% coverage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.transformers import rcs_basis, spline_columns

CRIMSON = "#E8175D"


def native_intercept(fitted) -> float:
    """Intercept expressed on the pre-scaling scale.

    Standardization folds the feature means into the intercept:
    b0 + sum_j beta_j (x_j - mu_j) / s_j  =  [b0 - sum_j beta_j mu_j / s_j] + ...

    Args:
        fitted: _FittedModel from develop_model() / final_model.joblib.

    Returns:
        Intercept to pair with the native-scale coefficients.
    """
    beta = fitted.model.coef_[0]
    return float(
        fitted.model.intercept_[0]
        - np.sum(beta * fitted.feature_means / fitted.feature_scales)
    )


def _spline_owner(fitted) -> dict[str, str]:
    """Map each spline basis column back to the predictor it expands."""
    return {
        name: col
        for col, knots in fitted.spline_knots.items()
        for name in spline_columns(col, len(knots) - 1)
    }


def _unit_labels(
    fitted,
    cfg_features: dict,
    unit_labels: dict[str, str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Label each feature's native unit and flag whether a native OR is meaningful.

    Args:
        fitted: _FittedModel whose feature_names give the encoded column order.
        cfg_features: pipeline_cfg["features"] block.
        unit_labels: Optional per-feature override of the default label. The
            default for an ordinal predictor is "per grade", which is wrong for
            one that counts things rather than grading them.

    Returns:
        (unit label per feature, boolean mask that is True where exp(beta/scale)
        is a clinically readable odds ratio).
    """
    continuous = set(cfg_features["continuous"])
    ordinal = set(cfg_features["ordinal"])
    overrides = unit_labels or {}
    owner = _spline_owner(fitted)
    units, readable = [], []
    for name in fitted.feature_names:
        if name in overrides:
            units.append(overrides[name])
            readable.append(True)
        elif name in owner:
            # A spline term has no standalone effect: the basis columns only mean
            # something together. Reported as a risk curve by spline_contrasts().
            units.append(f"spline basis for {owner[name]}")
            readable.append(False)
        elif name in continuous:
            units.append("per SD (Yeo-Johnson)")
            readable.append(False)
        elif name in ordinal:
            units.append("per grade")
            readable.append(True)
        else:
            units.append("present vs absent")
            readable.append(True)
    return units, np.array(readable)


def spline_contrasts(
    fitted,
    source: str,
    values: np.ndarray,
    reference: float,
    boot_coefs: np.ndarray | None = None,
    boot_scales: np.ndarray | None = None,
    boot_knots: np.ndarray | None = None,
) -> pd.DataFrame:
    """Odds ratios along a splined predictor, each against a fixed reference value.

    A splined predictor has no single odds ratio, so its effect is reported as a
    curve. Holding every other predictor fixed, the log-odds difference between
    value v and the reference is

        sum_j (beta_j / scale_j) * (basis_j(v) - basis_j(reference))

    over that predictor's basis columns, which is what this exponentiates.

    Bootstrap intervals are computed per iteration using that iteration's own
    knots and scaler SDs, since both are refitted on each resample.

    Args:
        fitted:      _FittedModel from final_model.joblib.
        source:      Splined predictor name, e.g. "LEEFTIJD_EXACT".
        values:      Values to report the odds ratio at.
        reference:   Value the odds ratios are taken against (OR = 1 there).
        boot_coefs:  Optional (B, n_features) standardized coefficients per iteration.
        boot_scales: Optional (B, n_features) Scaler SDs per iteration.
        boot_knots:  Optional (B, n_knots) knot locations per iteration.

    Returns:
        DataFrame with columns value, odds_ratio, and — when bootstrap inputs are
        given — odds_ratio_lo / odds_ratio_hi / n_bootstraps.
    """
    knots = fitted.spline_knots[source]
    idx = [fitted.feature_names.index(n)
           for n in spline_columns(source, len(knots) - 1)]
    values = np.asarray(values, dtype=float)

    def curve(coefs, scales, kn):
        delta = (rcs_basis(values, kn)
                 - rcs_basis(np.array([reference], dtype=float), kn))
        return np.exp(delta @ (coefs[idx] / scales[idx]))

    df = pd.DataFrame({
        "value": values,
        "odds_ratio": curve(fitted.model.coef_[0], fitted.feature_scales, knots),
    })

    if boot_coefs is not None and boot_scales is not None and boot_knots is not None:
        draws = np.vstack([
            curve(boot_coefs[b], boot_scales[b], boot_knots[b])
            for b in range(len(boot_coefs))
        ])
        df["odds_ratio_lo"] = np.percentile(draws, 2.5, axis=0)
        df["odds_ratio_hi"] = np.percentile(draws, 97.5, axis=0)
        df["n_bootstraps"] = len(boot_coefs)
    return df


def plot_spline_curve(df: pd.DataFrame, xlabel: str, reference: float, save_path=None):
    """Odds ratio against the reference value across a splined predictor's range.

    Args:
        df: Output of spline_contrasts() over a dense grid of values.
        xlabel: Axis label for the predictor, e.g. "Age (years)".
        reference: Reference value, marked on the plot where the OR is 1.
        save_path: Optional PNG destination.

    Returns:
        The created figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if "odds_ratio_lo" in df:
        ax.fill_between(df.value, df.odds_ratio_lo, df.odds_ratio_hi,
                        color=CRIMSON, alpha=0.18, label="95% bootstrap interval")
    ax.plot(df.value, df.odds_ratio, color=CRIMSON, linewidth=2)
    ax.axhline(1.0, color="0.35", linewidth=1, linestyle="--")
    ax.axvline(reference, color="0.6", linewidth=1, linestyle=":",
               label=f"Reference ({reference:g})")

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Odds ratio for 30-day mortality\n(vs {xlabel.split(' (')[0].lower()} {reference:g})")
    ax.grid(color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def coefficient_table(
    fitted,
    cfg_features: dict,
    boot_coefs: np.ndarray | None = None,
    boot_scales: np.ndarray | None = None,
    display_names: dict[str, str] | None = None,
    unit_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the reportable coefficient / odds-ratio table.

    Args:
        fitted: _FittedModel from final_model.joblib.
        cfg_features: pipeline_cfg["features"] block, used to label units.
        boot_coefs: Optional (B, n_features) standardized coefficients per
            bootstrap iteration; adds percentile CIs when given.
        boot_scales: Optional (B, n_features) Scaler SDs per iteration, required
            alongside boot_coefs to build native-scale CIs.
        display_names: Optional encoded name -> plot label mapping.
        unit_labels: Optional encoded name -> unit label override.

    Returns:
        DataFrame sorted by |standardized coefficient|, descending.
    """
    beta = fitted.model.coef_[0]
    units, readable = _unit_labels(fitted, cfg_features, unit_labels)
    native = beta / fitted.feature_scales

    df = pd.DataFrame({
        "feature": [(display_names or {}).get(n, n) for n in fitted.feature_names],
        "encoded_name": fitted.feature_names,
        "unit": units,
        "beta_std": beta,
        "or_per_sd": np.exp(beta),
        "beta_native": native,
        "or_native": np.where(readable, np.exp(native), np.nan),
    })

    if boot_coefs is not None:
        df["or_per_sd_lo"] = np.exp(np.percentile(boot_coefs, 2.5, axis=0))
        df["or_per_sd_hi"] = np.exp(np.percentile(boot_coefs, 97.5, axis=0))
        if boot_scales is not None:
            boot_native = boot_coefs / boot_scales
            lo = np.exp(np.percentile(boot_native, 2.5, axis=0))
            hi = np.exp(np.percentile(boot_native, 97.5, axis=0))
            df["or_native_lo"] = np.where(readable, lo, np.nan)
            df["or_native_hi"] = np.where(readable, hi, np.nan)
        df["n_bootstraps"] = len(boot_coefs)

    return df.reindex(
        df.beta_std.abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def _forest_columns(df: pd.DataFrame):
    """Pick the OR to plot per row: native where readable, per-SD otherwise.

    Returns:
        (estimate, lower, upper) arrays; lower/upper are None when the table
        carries no bootstrap CIs.
    """
    est = np.where(df.or_native.notna(), df.or_native, df.or_per_sd)
    if "or_native_lo" in df:
        lo = np.where(df.or_native.notna(), df.or_native_lo, df.or_per_sd_lo)
        hi = np.where(df.or_native.notna(), df.or_native_hi, df.or_per_sd_hi)
    else:
        lo = hi = None
    return est, lo, hi


def plot_coefficient_forest(df: pd.DataFrame, save_path=None, top_n: int | None = None,
                            display_names: dict[str, str] | None = None):
    """Forest plot of odds ratios on a log axis, ordered as the table is.

    Rows show the native-scale OR where that is a clinical unit and the per-SD OR
    for the Yeo-Johnson transformed continuous predictors. The y labels carry the
    predictor name only; the unit each row is expressed in stays in the `unit`
    column of coefficient_table(). Spline basis columns are dropped: a
    single row cannot represent them, and plot_spline_curve() shows them properly.
    A footnote names the predictors dropped that way, so the figure does not read
    as though they are absent from the model.

    Args:
        df: Output of coefficient_table().
        save_path: Optional PNG destination.
        top_n: Optionally restrict to the first N rows of the table.
        display_names: Optional predictor name -> label map, used for the footnote.

    Returns:
        The created figure.
    """
    dropped = df[df.unit.str.startswith("spline basis for")]
    df = df.drop(index=dropped.index)
    d = df.head(top_n) if top_n else df
    est, lo, hi = _forest_columns(d)
    ypos = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.6))
    if lo is not None:
        ax.hlines(ypos, lo, hi, color=CRIMSON, linewidth=1.6, alpha=0.85)
    ax.plot(est, ypos, "o", color=CRIMSON, markersize=6)
    ax.axvline(1.0, color="0.35", linewidth=1, linestyle="--")

    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels(d.feature, fontsize=10)
    ax.set_xlabel("Odds ratio for 30-day mortality (log scale, 95% bootstrap CI)")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Without this the figure gives no sign that a splined predictor is in the model
    # at all: it has no row here and its name appears nowhere on the axes.
    sources = sorted({u.removeprefix("spline basis for ") for u in dropped.unit})
    if sources:
        labels = ", ".join((display_names or {}).get(s, s) for s in sources)
        files = ", ".join(f"spline_{s}.png" for s in sources)
        ax.text(0, -0.5 / len(d) - 0.06,
                f"{labels}: fitted as a restricted cubic spline, so it has no single "
                f"odds ratio and is reported as a risk curve instead (see {files}).",
                transform=ax.transAxes, fontsize=8.5, color="0.35", va="top")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
