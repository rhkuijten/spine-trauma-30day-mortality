"""All model evaluation metrics for binary 30-day mortality classification.

Each function takes predicted probabilities and true binary labels and returns
a scalar or DataFrame of statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve as sk_roc_curve
from sklearn.calibration import calibration_curve

if TYPE_CHECKING:
    from src.validation.bootstrap import BootstrapResult


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tricube(d: np.ndarray) -> np.ndarray:
    return np.clip((1 - np.abs(d) ** 3) ** 3, 0, 1)


def calibration_lowess(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    frac: float = 0.75,
    n_grid: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Flexible calibration curve: LOWESS of the binary outcome on predicted risk.

    Every patient contributes at their own predicted probability. The previous
    construction binned into 10 equal-width bins first and smoothed the bin
    means; with 13% prevalence and right-skewed predictions most patients fell
    in the lowest bin and several upper bins were empty and silently dropped, so
    the curve rested on a handful of points of very unequal weight.

    Args:
        y_true: Binary event indicator array.
        y_prob: Predicted probabilities of the positive class.
        frac:   LOWESS bandwidth, fraction of the data in each local fit.
        n_grid: Number of evenly spaced points to evaluate the smoother on.

    Returns:
        (grid, smoothed observed risk) — both length n_grid, grid ascending and
        spanning the observed range of y_prob.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    grid = np.linspace(float(np.min(y_prob)), float(np.max(y_prob)), n_grid)
    smooth = lowess(y_true, y_prob, frac=frac, it=0, xvals=grid)
    return grid, np.asarray(smooth, dtype=float)


def _lowess(x: np.ndarray, y: np.ndarray, f: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS smoother with tricube weights. Returns (y_smooth, y_stderr)."""
    xwidth = f * (x.max() - x.min())
    N = len(x)
    order = np.argsort(x)
    y_sm = np.zeros_like(y, dtype=float)
    y_stderr = np.zeros_like(y, dtype=float)
    for i in range(N):
        dist = np.abs(x[order][i] - x[order]) / xwidth
        w = _tricube(dist)
        A = np.stack([w, x[order] * w]).T
        b = w * y[order]
        ATA = A.T @ A
        ATb = A.T @ b
        sol = np.linalg.solve(ATA, ATb)
        place = order[i]
        y_sm[place] = A[i] @ sol
        sigma2 = np.sum((A @ sol - y[order]) ** 2) / N
        y_stderr[place] = np.sqrt(sigma2 * A[i] @ np.linalg.inv(ATA) @ A[i])
    return y_sm, y_stderr




# ---------------------------------------------------------------------------
# Scalar metrics (used internally by bootstrap validation)
# ---------------------------------------------------------------------------

def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC-ROC. Higher is better; 0.5 = random, 1.0 = perfect.

    Args:
        y_true: Binary event indicator (1 = died within 30 days).
        y_prob: Predicted probabilities of the positive class.

    Returns:
        AUC-ROC scalar.
    """
    return float(roc_auc_score(y_true, y_prob))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score. Lower is better; 0.0 = perfect, 0.25 = no-skill baseline.

    Args:
        y_true: Binary event indicator (1 = died within 30 days).
        y_prob: Predicted probabilities of the positive class.

    Returns:
        Brier score scalar.
    """
    return float(brier_score_loss(y_true, y_prob))


def calibration_in_the_large(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calibration-in-the-large: intercept-only logistic GLM with logit(p) as offset.

    Fits logit(y) = alpha + offset(logit(p)), i.e. slope fixed at 1.
    Near 0 means mean predicted risk matches observed event rate.

    Uses sm.GLM with Binomial family, which correctly handles the offset
    argument (unlike sm.Logit's Newton optimizer).

    Args:
        y_true: Binary event indicator array.
        y_prob: Predicted probabilities of the positive class.

    Returns:
        Calibration-in-the-large scalar (alpha).
    """
    offset = _logit(y_prob)
    fit = sm.GLM(
        y_true,
        np.ones(len(y_true)),
        family=sm.families.Binomial(),
        offset=offset,
    ).fit(disp=False)
    return float(fit.params[0])


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calibration slope: slope from logit(y) ~ intercept + slope * logit(p).

    Near 1 means predictions are neither over- nor under-dispersed.

    Args:
        y_true: Binary event indicator array.
        y_prob: Predicted probabilities of the positive class.

    Returns:
        Calibration slope scalar.
    """
    X = sm.add_constant(_logit(y_prob))
    fit = sm.Logit(y_true, X).fit(disp=False)
    # add_constant may drop the constant when logit(p) is perfectly collinear;
    # in that degenerate case the fitted model has no slope to return.
    if len(fit.params) < 2:
        return float("nan")
    return float(fit.params[1])


def dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Decision curve analysis: net benefit across probability thresholds.

    Net benefit at threshold t:
        NB(t) = TP/N − FP/N × t/(1−t)
    Standardised NB divides by prevalence. It can become very negative at
    high thresholds when the false-positive penalty is large.

    Args:
        y_true:     Binary event indicator array.
        y_prob:     Predicted probabilities of the positive class.
        thresholds: Array of threshold values in (0, 1). Defaults to 0.01-0.99.

    Returns:
        DataFrame with columns:
            threshold, net_benefit, net_benefit_all,
            std_net_benefit, std_net_benefit_all.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    prev = float(np.mean(y_true))
    rows = []
    for t in thresholds:
        predicted_pos = y_prob >= t
        tp = float(np.sum(predicted_pos & (y_true == 1)))
        fp = float(np.sum(predicted_pos & (y_true == 0)))
        nb = tp / n - fp / n * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        std_nb = nb / prev if prev > 0 else 0.0
        std_nb_all = nb_all / prev if prev > 0 else 0.0
        rows.append({
            "threshold": t,
            "net_benefit": nb,
            "net_benefit_all": nb_all,
            "std_net_benefit": std_nb,
            "std_net_benefit_all": std_nb_all,
        })
    return pd.DataFrame(rows)


def dca_optout(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Opt-out DCA: net benefit of withholding surgery from high-mortality patients.

    At threshold t, withhold surgery from patients with p_mortality >= t.
    Reference: Wynants et al. 2019 (PMC6374190).

        NB(t) = TN/N − (1−t)/t × FN/N

    where TN = operated AND survived, FN = operated AND died.
    Standardised NB divides by survival rate (max possible TN benefit).
    Reference line "operate on all" = (1−ρ) − ρ × (1−t)/t.

    Args:
        y_true:     Binary mortality indicator (1 = died).
        y_prob:     Predicted probability of mortality.
        thresholds: Mortality thresholds in (0, 1). Defaults to 0.10–0.50.

    Returns:
        DataFrame with columns:
            threshold, net_benefit, net_benefit_operate_all,
            std_net_benefit, std_net_benefit_operate_all.
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.99, 90)
    n = len(y_true)
    prev = float(np.mean(y_true))
    surv_rate = 1 - prev
    rows = []
    for t in thresholds:
        operated = y_prob < t
        tn = float(np.sum(operated & (y_true == 0)))
        fn = float(np.sum(operated & (y_true == 1)))
        nb = tn / n - (1 - t) / t * fn / n
        nb_all = surv_rate - prev * (1 - t) / t
        std_nb = nb / surv_rate if surv_rate > 0 else 0.0
        std_nb_all = nb_all / surv_rate if surv_rate > 0 else 0.0
        rows.append({
            "threshold": t,
            "net_benefit": nb,
            "net_benefit_operate_all": nb_all,
            "std_net_benefit": std_nb,
            "std_net_benefit_operate_all": std_nb_all,
        })
    return pd.DataFrame(rows)


def summarise(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """One-row DataFrame with AUC, Brier, calibration-in-the-large and slope.

    Args:
        y_true: Binary event indicator array.
        y_prob: Predicted probabilities of the positive class.

    Returns:
        DataFrame with columns
        ['auc', 'brier', 'calibration_in_the_large', 'calibration_slope'].
    """
    return pd.DataFrame({
        "auc":                      [auc(y_true, y_prob)],
        "brier":                    [brier(y_true, y_prob)],
        "calibration_in_the_large": [calibration_in_the_large(y_true, y_prob)],
        "calibration_slope":        [calibration_slope(y_true, y_prob)],
    })


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    boot_result: BootstrapResult | None = None,
    n_boot_band: int = 200,
) -> plt.Figure:
    """LOWESS calibration curve with prediction histograms.

    Always plots apparent predictions. If boot_result is provided, the
    optimism-corrected calibration-in-the-large and slope are annotated.

    Args:
        y_true:      Binary event indicator array.
        y_prob:      Apparent predicted probabilities (from the full-data model).
        boot_result: BootstrapResult for corrected metric annotation only.
                     Probabilities are never transformed.
        n_boot_band: Resamples used for the pointwise 95% band around the curve.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots()

    if boot_result is not None:
        cal_i = boot_result.corrected_metrics["calibration_in_the_large"]
        cal_s = boot_result.corrected_metrics["calibration_slope"]
        annotation = (
            f"CITL = {cal_i:.2f}, Slope = {cal_s:.2f} (optimism-corrected)"
        )
        title_suffix = " (apparent predictions; corrected metrics annotated)"
    else:
        cal_i = calibration_in_the_large(y_true, y_prob)
        cal_s = calibration_slope(y_true, y_prob)
        annotation = f"CITL = {cal_i:.2f}, Slope = {cal_s:.2f}"
        title_suffix = " (apparent)"

    grid, curve = calibration_lowess(y_true, y_prob)

    # Pointwise band from resampling the cohort with the model held fixed, on the
    # same grid the curve is drawn on.
    rng = np.random.default_rng(42)
    n = len(y_true)
    band = np.empty((n_boot_band, len(grid)))
    for b in range(n_boot_band):
        idx = rng.integers(0, n, n)
        band[b] = np.interp(
            grid, *calibration_lowess(y_true[idx], y_prob[idx])
        )
    lo = np.percentile(band, 2.5, axis=0)
    hi = np.percentile(band, 97.5, axis=0)

    ax.plot(grid, curve, color="black", linewidth=1.5, label=annotation)
    ax.fill_between(grid, lo, hi, alpha=0.3, color="grey",
                    label="95% pointwise band")
    ax.plot([0, 1], [0, 1], color="red", linewidth=0.8,
            label="Perfect calibration")

    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    n_total = len(y_prob)
    ax.hist(pos_probs, weights=np.ones_like(pos_probs) / n_total,
            bins=50, alpha=0.4, color="red", label="Died within 30 days")
    ax.hist(neg_probs, weights=-np.ones_like(neg_probs) / n_total,
            bins=50, alpha=0.4, color="green", label="Survived")

    ax.set_xlabel("Predicted probability", size=12)
    ax.set_ylabel("Observed proportion", size=12)
    ax.set_title(f"Calibration{title_suffix}", pad=25, size=15, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    boot_result: BootstrapResult | None = None,
) -> plt.Figure:
    """ROC curve with AUC label; reports optimism-corrected AUC if available.

    The curve shape uses the apparent predictions (you cannot meaningfully
    reshape a ROC curve after optimism correction). The AUC in the legend
    shows the corrected value with 95% CI when boot_result is provided.

    Args:
        y_true:      Binary event indicator array.
        y_prob:      Apparent predicted probabilities.
        boot_result: BootstrapResult for corrected AUC and CI. If None,
                     the apparent AUC is shown.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots()
    fpr, tpr, _ = sk_roc_curve(y_true, y_prob)

    if boot_result is not None:
        auc_val = boot_result.corrected_metrics["auc"]
        ci_lo = boot_result.ci_lower["auc"]
        ci_hi = boot_result.ci_upper["auc"]
        label = f"Model (AUC = {auc_val:.2f}, 95% CI {ci_lo:.2f}–{ci_hi:.2f})"
    else:
        auc_val = auc(y_true, y_prob)
        label = f"Model (AUC = {auc_val:.2f})"

    ax.plot(fpr, tpr, color="red", label=label)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="No skill")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", size=12)
    ax.set_ylabel("True Positive Rate", size=12)
    ax.set_title("Discrimination", pad=25, size=15, fontweight="bold")
    ax.grid(visible=True, which="major")
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> plt.Figure:
    """Decision Curve Analysis plot using standardised net benefit.

    Standardised NB = NB / prevalence, making models comparable across
    populations with different event rates. The full threshold range can
    produce very negative treat-all net benefit near 1.0, so the plot uses a
    symmetric-log y-axis when needed rather than clipping the curve.
    Always uses apparent predictions; optimism-corrected performance estimates
    are reported separately as scalar metrics, not by transforming probabilities.

    Args:
        y_true:     Binary event indicator array.
        y_prob:     Apparent predicted probabilities.
        thresholds: Threshold array in (0, 1). Defaults to 0.01-0.99.

    Returns:
        Matplotlib Figure.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    title_suffix = " (apparent)"
    df = dca(y_true, y_prob, thresholds=thresholds)

    fig, ax = plt.subplots()
    ax.plot(df["threshold"], df["std_net_benefit"],
            color="blue", label="Model")
    ax.plot(df["threshold"], df["std_net_benefit_all"],
            color="grey", label="Treat all")
    ax.plot([0, 1], [0, 0], color="grey", linestyle=":", label="Treat none")

    y_values = np.concatenate([
        df["std_net_benefit"].to_numpy(),
        df["std_net_benefit_all"].to_numpy(),
        np.array([0.0]),
    ])
    y_values = y_values[np.isfinite(y_values)]
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if y_min < -1:
        ax.set_yscale("symlog", linthresh=0.05)
        y_label = "Standardised Net Benefit (symlog scale)"
    else:
        y_label = "Standardised Net Benefit"
    pad = max(0.05, 0.05 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Threshold Probability", size=12)
    ax.set_ylabel(y_label, size=12)
    ax.set_title(f"Decision Curve Analysis{title_suffix}",
                 pad=25, size=15, fontweight="bold")
    ax.grid(which="major")
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
