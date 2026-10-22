"""SHAP plots: global importance bar, beeswarm, and per-patient waterfall.

The global plots are on the log-odds scale, where the linear model's SHAP values
are exact and additive. The waterfall reports absolute risk instead, so that a
clinician reads percentage points rather than log-odds; see
_probability_contributions() for how the values are put on that scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

CRIMSON = "#E8175D"
_XLABEL = "Impact on predicted log-odds of 30-day mortality"
_LABEL_SIZE = 11

# Waterfall palette: raises risk / lowers risk.
_UP, _DOWN = "#F08080", "#2E8B2E"


def _normalize_fonts(ax) -> None:
    """Force one label size on an axes; shap's own plots hard-code a larger one."""
    ax.tick_params(labelsize=_LABEL_SIZE)
    ax.xaxis.label.set_size(_LABEL_SIZE)
    ax.yaxis.label.set_size(_LABEL_SIZE)


def _save(fig, save_path: Path | None) -> None:
    """Write fig to save_path (creating parents) and close it, or leave it open."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_importance_bar(expl, ax=None, top_n: int = 15, save_path=None):
    """Horizontal bar chart of mean |SHAP| per feature (panel A of the summary).

    Args:
        expl: Explanation from compute_shap().
        ax: Optional axes to draw into; a new figure is created when None.
        top_n: Number of highest-ranked features to show.
        save_path: Optional PNG destination (only used when ax is None).

    Returns:
        The axes drawn into.
    """
    imp = np.abs(expl.values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top_n][::-1]  # ascending: barh draws bottom-up
    names = [expl.feature_names[i] for i in order]
    vals = imp[order]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.38 * len(order) + 1.2))

    ax.barh(names, vals, color=CRIMSON, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + vals.max() * 0.015, y, f"{v:.2f}",
                va="center", fontsize=9, color=CRIMSON)

    ax.set_xlim(0, vals.max() * 1.14)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _normalize_fonts(ax)

    if fig is not None:
        _save(fig, save_path)
    return ax


def plot_beeswarm(expl, ax=None, top_n: int = 15, save_path=None):
    """Beeswarm of per-patient SHAP values coloured by feature value (panel B).

    Features are ordered by mean |SHAP|, matching plot_importance_bar().

    Args:
        expl: Explanation from compute_shap().
        ax: Optional axes to draw into; a new figure is created when None.
        top_n: Number of highest-ranked features to show.
        save_path: Optional PNG destination (only used when ax is None).

    Returns:
        The axes drawn into.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.38 * min(top_n, len(expl.feature_names)) + 1.2))

    shap.plots.beeswarm(
        expl, max_display=top_n, ax=ax, show=False, plot_size=None,
        group_remaining_features=False,
        color_bar_label="Predictor input value",
    )
    ax.set_xlabel(_XLABEL)
    _normalize_fonts(ax)

    if fig is not None:
        _save(fig, save_path)
    return ax


def plot_global_summary(expl, top_n: int = 15, save_path=None):
    """Two-panel figure: A = mean |SHAP| bar chart, B = beeswarm.

    Args:
        expl: Explanation from compute_shap().
        top_n: Number of highest-ranked features to show in both panels.
        save_path: Optional PNG destination.

    Returns:
        The created figure.
    """
    n_rows = min(top_n, len(expl.feature_names))
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 2 * (0.38 * n_rows + 1.4)),
    )
    plot_importance_bar(expl, ax=axes[0], top_n=top_n)
    plot_beeswarm(expl, ax=axes[1], top_n=top_n)
    for ax, letter in zip(axes, "AB"):
        ax.set_title(letter, loc="center", fontweight="bold", fontsize=13)
    fig.tight_layout(h_pad=2.0)

    # The beeswarm reserves room for its colour bar, so panel B ends up narrower
    # than panel A; match A to B so the two feature-name columns line up.
    pos_a, pos_b = axes[0].get_position(), axes[1].get_position()
    axes[0].set_position([pos_b.x0, pos_a.y0, pos_b.width, pos_a.height])

    _save(fig, save_path)
    return fig


def _shown_value(value) -> str:
    """Format one predictor value for a row label: a number, or a category level."""
    return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)


def _probability_contributions(expl, index: int):
    """Re-express one patient's log-odds SHAP values as percentage points of risk.

    SHAP is additive on the log-odds scale, not on the probability scale, so the
    values are rescaled by the constant (p_pred - p_base) / sum(phi) to make them
    chain from the base rate to the prediction. The sigmoid is monotone, so that
    factor is always positive: every contribution keeps its SHAP sign and the
    ranking is unchanged. The rescaled numbers are an allocation of the total risk
    difference, not each predictor's marginal effect in isolation.

    Args:
        expl: Explanation from compute_shap().
        index: Row position of the patient within expl.

    Returns:
        (contributions in percentage points, base probability, predicted probability).
    """
    phi = np.asarray(expl.values[index], dtype=float)
    base = float(expl.base_values[index])
    p_base = 1.0 / (1.0 + np.exp(-base))
    p_pred = 1.0 / (1.0 + np.exp(-(base + phi.sum())))

    total = phi.sum()
    share = phi / total if total != 0 else np.zeros_like(phi)
    return share * (p_pred - p_base) * 100.0, p_base, p_pred


def _waterfall_rows(pp, names, data, top_n: int, group_threshold_pct: float | None):
    """Pool small signed contributions, retaining the exact total risk difference.

    A threshold replaces the top-n limit: every predictor whose share of the
    patient's summed absolute contributions is above the threshold is retained,
    even if that yields more than top_n individual bars. The share is relative so
    a low-risk patient, whose contributions are all small in percentage points,
    is not reduced to one pooled bar. Pooling uses signed values.
    """
    order = np.argsort(-np.abs(pp))
    if group_threshold_pct is None:
        shown, pooled = order[:top_n], order[top_n:]
    else:
        if not np.isfinite(group_threshold_pct) or group_threshold_pct < 0:
            raise ValueError("group_threshold_pct must be finite and non-negative")
        total = np.abs(pp).sum()
        share = 100 * np.abs(pp[order]) / total if total else np.zeros(len(pp))
        individual = share > group_threshold_pct
        shown, pooled = order[individual], order[~individual]
    rows = [(f"{_shown_value(data[i])} = {names[i]}", pp[i]) for i in shown]
    if len(pooled):
        rows.append((f"Other predictors (n = {len(pooled)})", pp[pooled].sum()))
    return rows


def _chevron(ax, x0: float, x1: float, y: float, color: str, height: float = 0.62):
    """Draw one waterfall bar from x0 to x1, pointed in its direction of travel."""
    tip = min(abs(x1 - x0) * 0.35, np.ptp(ax.get_xlim()) * 0.012)
    step = np.sign(x1 - x0) * tip
    pts = [(x0, y - height / 2), (x1 - step, y - height / 2), (x1, y),
           (x1 - step, y + height / 2), (x0, y + height / 2)]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=color, edgecolor="none"))


def _waterfall_xlim(rows, p_base: float, p_pred: float) -> tuple[float, float]:
    """Axis limits covering the whole chain plus a margin for the outside labels."""
    cursor, seen = p_base, [p_base, p_pred]
    for _, value in rows[::-1]:
        cursor += value / 100.0
        seen.append(cursor)
    lo, hi = min(seen), max(seen)
    pad = max((hi - lo) * 0.12, 0.02)
    # Intermediate sums of the allocated contributions are not predictions and
    # can cross 0 or 1. Keep those bars visible instead of clipping their tips.
    left = max(0.0, lo - pad) if lo >= 0 else lo - pad
    right = min(1.0, hi + pad) if hi <= 1 else hi + pad
    return left, right


def _bar_label(ax, x0: float, x1: float, y: float, value: float, color: str) -> None:
    """Label contributions as absolute risk change in %, rounded only for display."""
    text = f"{value:+.1f}%" if round(value, 1) else "0.0%"
    if abs(x1 - x0) > np.ptp(ax.get_xlim()) * 0.12:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center",
                color="white", fontsize=10)
    else:
        # Keep thin-bar labels to the right so they cannot overlap predictor names.
        outside = np.ptp(ax.get_xlim()) * 0.012
        ax.text(max(x0, x1) + outside, y, text, va="center", fontsize=10,
                color=color, ha="left")


def _waterfall_axes(ax, rows, p_base: float, p_pred: float) -> None:
    """Apply the shared waterfall styling: tick labels, guides and spines."""
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([t for t, _ in rows][::-1], fontsize=_LABEL_SIZE)
    ax.set_ylim(-1.0, len(rows) - 0.3)  # room below the bars for the reference label
    for line in (p_base, p_pred):
        ax.axvline(line, color="0.6", linewidth=0.9, linestyle=(0, (2, 3)), zorder=0)
    ax.text(p_pred, len(rows) - 0.45, f"{p_pred:.1%}", ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    # The chain's starting point; the white box keeps the dashed line off the text.
    ax.text(p_base, -0.45, f"Reference {p_base:.1%}", ha="center", va="top",
            fontsize=_LABEL_SIZE, color="0.3",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Predicted 30-day mortality risk")
    ax.grid(axis="y", color="0.92", linewidth=0.8, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _normalize_fonts(ax)


def plot_waterfall(expl, index: int, top_n: int = 15, save_path=None,
                   group_threshold_pct: float | None = None):
    """Waterfall showing how each predictor moves one patient from base to prediction.

    Contributions are shown as percentage points of absolute risk, chaining
    from the SHAP reference probability at the bottom to this patient's prediction
    at the top. See _probability_contributions() for how the log-odds SHAP values
    are put on that scale. Labels are rounded to one decimal; bars use unrounded
    values.

    Args:
        expl: Explanation from compute_shap().
        index: Row position of the patient within expl.
        top_n: Number of features to show individually when no threshold is set.
        save_path: Optional PNG destination.
        group_threshold_pct: Pool predictors whose absolute contribution is at
            or below this percentage of the patient's summed absolute
            contributions into one signed net contribution. When set, this
            replaces the top_n limit.

    Returns:
        The created figure.
    """
    pp, p_base, p_pred = _probability_contributions(expl, index)
    names = list(expl.feature_names)
    data = np.asarray(expl.data[index])

    rows = _waterfall_rows(pp, names, data, top_n, group_threshold_pct)

    fig, ax = plt.subplots(figsize=(10, 0.46 * len(rows) + 2.0))
    ax.set_xlim(*_waterfall_xlim(rows, p_base, p_pred))

    cursor = p_base
    for y, (_, value) in enumerate(rows[::-1]):
        color = _UP if value >= 0 else _DOWN
        _chevron(ax, cursor, cursor + value / 100.0, y, color)
        _bar_label(ax, cursor, cursor + value / 100.0, y, value, color)
        cursor += value / 100.0

    _waterfall_axes(ax, rows, p_base, p_pred)
    _save(fig, save_path)
    return fig
