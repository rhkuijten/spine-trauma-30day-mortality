"""Riley 2023 stability measures for clinical prediction models.

Computes prediction instability (MAPE), calibration instability,
DCA instability, and predictor selection frequency from bootstrap iterations.

Reference:
    Riley RD, Collins GS. Stability of clinical prediction models developed
    using statistical or machine learning methods. Biometrical Journal. 2023.
    PMC10952221.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics.metrics import calibration_lowess, dca, dca_optout, _lowess

_AGE_BINS = [0, 65, 80, np.inf]
_AGE_LABELS = ["<65", "65–80", "≥80"]


@dataclasses.dataclass
class StabilityResult:
    """Output of compute_stability().

    Attributes:
        mape_per_individual: MAPE_i for each patient in the original dataset.
        overall_mape:        Mean of mape_per_individual.
        predictor_freq:      Dict mapping feature name → fraction of bootstrap
                             iterations in which that feature had non-zero coef.
        boot_preds:          Array (B, n) of per-iteration predictions on original data.
        boot_params:         List of best (C, l1_ratio) per iteration.
        p_apparent:          Predictions from the apparent (full-data) model, shape (n,).
    """
    mape_per_individual: np.ndarray
    overall_mape: float
    predictor_freq: dict[str, float]
    boot_preds: np.ndarray
    boot_params: list[dict]
    p_apparent: np.ndarray


def compute_stability(
    p_apparent: np.ndarray,
    boot_preds: np.ndarray,
    boot_coef_masks: list[np.ndarray],
    feature_names: list[str],
    boot_params: list[dict],
) -> StabilityResult:
    """Compute all stability measures from bootstrap predictions.

    Args:
        p_apparent:      Shape (n,) — original model predictions on full data.
        boot_preds:      Shape (B, n) — each bootstrap model's predictions on full data.
        boot_coef_masks: List of B boolean arrays, True where |coef| > 0.
        feature_names:   Feature names corresponding to coef positions.
        boot_params:     List of B dicts with best_params per iteration.

    Returns:
        StabilityResult with all stability measures populated.
    """
    mape_per_individual = np.mean(
        np.abs(boot_preds - p_apparent[np.newaxis, :]), axis=0
    )
    overall_mape = float(np.mean(mape_per_individual))

    masks_arr = np.stack(boot_coef_masks, axis=0)  # (B, n_features)
    predictor_freq = {
        name: float(np.mean(masks_arr[:, i]))
        for i, name in enumerate(feature_names)
    }

    return StabilityResult(
        mape_per_individual=mape_per_individual,
        overall_mape=overall_mape,
        predictor_freq=predictor_freq,
        boot_preds=boot_preds,
        boot_params=boot_params,
        p_apparent=p_apparent,
    )


def predictor_frequency_table(result: StabilityResult) -> pd.DataFrame:
    """Return predictor selection frequencies sorted descending.

    Args:
        result: StabilityResult from compute_stability().

    Returns:
        DataFrame with columns ['predictor', 'selection_frequency'].
    """
    rows = sorted(result.predictor_freq.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(rows, columns=["predictor", "selection_frequency"])


def plot_prediction_instability(
    result: StabilityResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter of all bootstrap predictions vs the apparent prediction (Riley Level 4).

    For each individual, all B bootstrap predictions are plotted against that
    individual's apparent predicted probability. The diagonal y=x line
    represents perfect stability. The spread of points above/below the
    diagonal shows how much each individual's risk estimate varies across
    resampled models.

    Args:
        result:    StabilityResult from compute_stability().
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    n_boot = result.boot_preds.shape[0]
    x = np.tile(result.p_apparent, n_boot)
    y = result.boot_preds.ravel()

    ax.scatter(x, y, alpha=0.05, s=4, color="steelblue", rasterized=True)
    ax.plot([0, 1], [0, 1], color="red", linewidth=1.5, label="Perfect stability")

    # LOWESS-smoothed 2.5/97.5 percentile envelope (Collins et al. 2023)
    lower = np.percentile(result.boot_preds, 2.5, axis=0)
    upper = np.percentile(result.boot_preds, 97.5, axis=0)
    p = result.p_apparent
    order = np.argsort(p)
    lower_sm, _ = _lowess(p[order], lower[order], f=0.2)
    upper_sm, _ = _lowess(p[order], upper[order], f=0.2)
    ax.plot(p[order], lower_sm, color="grey", linewidth=1.2, linestyle="--",
            label="Bootstrap 95% interval (smoothed)")
    ax.plot(p[order], upper_sm, color="grey", linewidth=1.2, linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (apparent model)", size=12)
    ax.set_ylabel("Predicted probability (bootstrap model)", size=12)
    ax.set_title("Prediction instability plot", pad=15, size=14, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mape_instability(
    result: StabilityResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter of individual MAPE vs apparent predicted probability (Riley Level 4).

    Shows where across the risk spectrum prediction instability is greatest.
    A horizontal dashed line marks the overall mean MAPE.

    Args:
        result:    StabilityResult from compute_stability().
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result.p_apparent, result.mape_per_individual,
               alpha=0.4, s=15, color="steelblue")
    ax.axhline(result.overall_mape, color="red", linestyle="--",
               label=f"Mean MAPE = {result.overall_mape:.3f}")
    ax.set_xlabel("Predicted probability (apparent model)", size=12)
    ax.set_ylabel("Individual MAPE", size=12)
    ax.set_title("MAPE instability plot", pad=15, size=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_subgroup_instability(
    result: StabilityResult,
    X: pd.DataFrame,
    sex_col: str = "Geslacht01",
    age_col: str = "LEEFTIJD_EXACT",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """MAPE instability plots split by sex and age group (Riley Level 3).

    Two panels side by side:
      - Left:  individual MAPE coloured by sex (male / female).
      - Right: individual MAPE coloured by age group (<65 / 65–80 / ≥80).

    Missing values in sex or age are excluded from the respective panel.
    The overall mean MAPE line is shown in both panels for reference.

    Args:
        result:   StabilityResult from compute_stability().
        X:        Original feature DataFrame aligned with result.p_apparent.
        sex_col:  Column name for sex (0 = male, 1 = female; verified against
                  the raw MAN_VROUW column — see EDA section 2g).
        age_col:  Column name for age in years (continuous).
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    mape = result.mape_per_individual
    p = result.p_apparent

    # ── Left panel: sex ──────────────────────────────────────────────────────
    ax = axes[0]
    if sex_col in X.columns:
        sex = X[sex_col].to_numpy(dtype=float)
        for val, label, color in [(0, "Male", "steelblue"), (1, "Female", "tomato")]:
            mask = sex == val
            if mask.any():
                grp_mape = float(np.mean(mape[mask]))
                ax.scatter(p[mask], mape[mask], alpha=0.5, s=15, color=color,
                           label=f"{label} (n={mask.sum()}, mean MAPE={grp_mape:.3f})")
                ax.axhline(grp_mape, color=color, linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, f"Column '{sex_col}' not found",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Predicted probability (apparent model)", size=11)
    ax.set_ylabel("Individual MAPE", size=11)
    ax.set_title("Subgroup instability — sex", size=13, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Right panel: age group ────────────────────────────────────────────────
    ax = axes[1]
    if age_col in X.columns:
        age = X[age_col].to_numpy(dtype=float)
        age_groups = pd.cut(age, bins=_AGE_BINS, labels=_AGE_LABELS, right=False)
        colors = ["#2196F3", "#FF9800", "#9C27B0"]
        for label, color in zip(_AGE_LABELS, colors):
            mask = np.array(age_groups == label)
            if mask.any():
                grp_mape = float(np.mean(mape[mask]))
                ax.scatter(p[mask], mape[mask], alpha=0.5, s=15, color=color,
                           label=f"{label} (n={mask.sum()}, mean MAPE={grp_mape:.3f})")
                ax.axhline(grp_mape, color=color, linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, f"Column '{age_col}' not found",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Predicted probability (apparent model)", size=11)
    ax.set_title("Subgroup instability — age group", size=13, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Subgroup prediction instability",
                 size=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _add_hanging_histogram(
    ax: plt.Axes,
    p: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 50,
) -> None:
    """Overlay a hanging histogram on a calibration axes.

    Deaths (y=1) hang upward in red; survivors (y=0) hang downward in green.
    Weights are normalised by total N so bars stay within the [0, 1] y-range.
    """
    p_pos = p[y_true == 1]
    p_neg = p[y_true == 0]
    w_pos =  np.ones_like(p_pos) / len(p)
    w_neg = -np.ones_like(p_neg) / len(p)
    ax.hist(p_pos, weights=w_pos, bins=n_bins, alpha=0.4,
            color="tomato", label="Death")
    ax.hist(p_neg, weights=w_neg, bins=n_bins, alpha=0.4,
            color="mediumseagreen", label="Alive")


def plot_calibration_instability(
    y_true: np.ndarray,
    p_apparent: np.ndarray,
    result: StabilityResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay LOWESS calibration curves from all bootstrap models on the apparent curve.

    Each curve is a LOWESS of the observed outcome on that model's predicted risk,
    fitted on the raw patient-level data rather than on binned means, so every
    patient contributes at their own predicted probability. All curves are
    evaluated against the same y_true and drawn on the same grid, making them
    directly comparable to the apparent model curve (red). This answers the
    instability question: how much does calibration vary when the model is refit
    on resampled data? An outcome histogram is shown below the main panel.

    Args:
        y_true:     True binary outcomes.
        p_apparent: Original model predicted probabilities, shape (n,).
        result:     StabilityResult from compute_stability().
        save_path:  Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, zorder=3,
            label="Perfect calibration")

    for b in range(result.boot_preds.shape[0]):
        grid_b, curve_b = calibration_lowess(y_true, result.boot_preds[b])
        ax.plot(grid_b, curve_b, color="lightgrey", alpha=0.3, linewidth=0.8,
                zorder=2)

    grid, curve = calibration_lowess(y_true, p_apparent)
    ax.plot(grid, curve, color="red", linewidth=2, zorder=4,
            label="Apparent model")

    _add_hanging_histogram(ax, p_apparent, y_true)
    ax.set_xlabel("Mean predicted probability", size=12)
    ax.set_ylabel("Observed fraction of positives", size=12)
    ax.set_title("Calibration", pad=15, size=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_dca_instability(
    y_true: np.ndarray,
    p_apparent: np.ndarray,
    result: StabilityResult,
    thresholds: np.ndarray | None = None,
    opt_out: bool = False,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay decision curves from all bootstrap models on the original curve.

    All bootstrap DCA curves are drawn in light grey. A 95% stability band
    (2.5th-97.5th percentile across all iterations) is filled behind the main
    curve.

    When opt_out=True, uses the opt-out DCA formula (Wynants et al. 2019):
        NB = TN/N − (1−t)/t × FN/N
    where t is the mortality probability threshold for withholding surgery.
    The reference line is "operate on all" rather than "treat none".

    Args:
        y_true:     True binary mortality outcomes (1 = died).
        p_apparent: Original model predicted probabilities of mortality, shape (n,).
        result:     StabilityResult from compute_stability().
        thresholds: Threshold array. Defaults to 0.10–0.50 (opt-out) or 0.01–0.99.
        opt_out:    If True, use opt-out DCA framing for withholding surgery.
        save_path:  Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if opt_out:
        if thresholds is None:
            thresholds = np.linspace(0.10, 0.99, 90)

        boot_nb: list[np.ndarray] = []
        for b in range(result.boot_preds.shape[0]):
            df_b = dca_optout(y_true, result.boot_preds[b], thresholds=thresholds)
            nb_b = df_b["std_net_benefit"].values
            boot_nb.append(nb_b)
            ax.plot(df_b["threshold"], nb_b, color="lightgrey", alpha=0.3, linewidth=0.8)

        df_app = dca_optout(y_true, p_apparent, thresholds=thresholds)
        ax.plot(df_app["threshold"], df_app["std_net_benefit"],
                color="red", linewidth=2, label="Original model")
        ax.plot(df_app["threshold"], df_app["std_net_benefit_operate_all"],
                color="gray", linewidth=1.5, linestyle="--", label="Operate on all")
        ax.axhline(0, color="black", linewidth=1, label="Withhold from all")

        x_label = "Threshold probability of 30-day mortality"
        title = "Opt-out decision curve instability plot"
    else:
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        boot_nb = []
        for b in range(result.boot_preds.shape[0]):
            df_b = dca(y_true, result.boot_preds[b], thresholds=thresholds)
            nb_b = df_b["std_net_benefit"].values
            boot_nb.append(nb_b)
            ax.plot(df_b["threshold"], nb_b, color="lightgrey", alpha=0.3, linewidth=0.8)

        df_app = dca(y_true, p_apparent, thresholds=thresholds)
        ax.plot(df_app["threshold"], df_app["std_net_benefit"],
                color="red", linewidth=2, label="Original model")
        ax.plot(df_app["threshold"], df_app["std_net_benefit_all"],
                color="gray", linewidth=1.5, linestyle="--", label="Treat all")
        ax.axhline(0, color="black", linewidth=1, label="Treat none")

        x_label = "Threshold probability"
        title = "Decision curve instability plot"

    nb_arr = np.stack(boot_nb, axis=0)
    lo = np.percentile(nb_arr, 2.5, axis=0)
    hi = np.percentile(nb_arr, 97.5, axis=0)
    ax.fill_between(thresholds, lo, hi, color="lightgrey", alpha=0.25,
                    label="Bootstrap 95% stability interval")

    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Standardised net benefit")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_instability(
    y_true: np.ndarray,
    p_apparent: np.ndarray,
    result: StabilityResult,
    corrected_auc: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """ROC curves from all bootstrap models overlaid on the apparent model curve.

    Bootstrap ROC curves are drawn in light grey. The apparent model curve is
    shown in red. AUC values are reported in the legend; if corrected_auc is
    supplied it is shown alongside the apparent AUC.

    Note: the ROC curve shape is invariant to monotone probability
    transformations, so there is no separate 'corrected curve' — only the
    corrected AUC scalar differs from the apparent AUC.

    Args:
        y_true:        True binary outcomes.
        p_apparent:    Original model predicted probabilities, shape (n,).
        result:        StabilityResult from compute_stability().
        corrected_auc: Optimism-corrected AUC scalar to display in the legend.
        save_path:     Optional file path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    from sklearn.metrics import auc as sk_auc, roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="No discrimination")

    for b in range(result.boot_preds.shape[0]):
        fpr_b, tpr_b, _ = roc_curve(y_true, result.boot_preds[b])
        ax.plot(fpr_b, tpr_b, color="lightgrey", alpha=0.3, linewidth=0.8)

    fpr, tpr, _ = roc_curve(y_true, p_apparent)
    apparent_auc = sk_auc(fpr, tpr)
    if corrected_auc is not None:
        auc_label = f"Apparent model (AUC={apparent_auc:.3f}, corrected={corrected_auc:.3f})"
    else:
        auc_label = f"Apparent model (AUC={apparent_auc:.3f})"
    ax.plot(fpr, tpr, color="red", linewidth=2, label=auc_label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 – Specificity", size=12)
    ax.set_ylabel("Sensitivity", size=12)
    ax.set_title("Discrimination", pad=15, size=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
