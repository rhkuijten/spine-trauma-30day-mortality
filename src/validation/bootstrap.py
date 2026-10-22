"""Bootstrap validation with full model redevelopment per iteration.

Each bootstrap iteration repeats the entire model development process:
  preprocessing (refit) → GridSearchCV tuning → final LogisticRegression fit.

This follows Harrell's optimism-correction method:
  optimism = in-bag performance − performance on original data
  corrected = apparent − mean(optimism)

Stability measures (Riley 2023) are derived from per-iteration predictions
on the original dataset.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.utils.parallel import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from src.metrics.metrics import (
    auc,
    brier,
    calibration_in_the_large,
    calibration_slope,
)
from src.preprocessing.pipeline import build_preprocessing_pipeline

if TYPE_CHECKING:
    from src.metrics.stability import StabilityResult


# ---------------------------------------------------------------------------
# Internal fitted-model container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FittedModel:
    pipeline: Pipeline           # full fitted sklearn Pipeline (preprocessor + logreg)
    preprocessor: object         # pipeline[:-1]: fitted preprocessing sub-pipeline
    model: LogisticRegression    # pipeline[-1]: fitted LogisticRegression
    best_params: dict
    non_zero_mask: np.ndarray   # bool array over post-encoding feature names
    feature_names: list[str]
    feature_means: np.ndarray   # Scaler mean_ per feature, on the pre-scaling scale
    feature_scales: np.ndarray  # Scaler scale_ per feature; beta / scale_ un-standardizes
    spline_knots: dict          # {source col: knot array} for spline predictors; {} if none

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_pp = self.preprocessor.transform(X)
        return self.model.predict_proba(X_pp)[:, 1]


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BootstrapResult:
    """Full output of bootstrap_validate().

    Attributes:
        apparent_metrics:   Metrics on the original full dataset (optimistic).
        corrected_metrics:  Apparent metrics minus mean optimism. Report these.
        ci_lower:           Location-shifted bootstrap interval for the corrected metric
                            (Noma et al. 2021): the percentile interval of
                            apparent_on_resample, shifted by the mean optimism. Covers
                            sampling variability on this cohort, not uncertainty in the
                            optimism correction.
        ci_upper:           Upper bound of the same.
        corrected_per_iter: Per-iteration corrected values: {metric: [apparent − optimism_b, ...]}.
                            Raw record of how far the correction moved per resample.
        apparent_on_resample: {metric: [...]} — the apparent model, held fixed, scored on
                            each iteration's bootstrap resample. Source of ci_lower/ci_upper.
        stability:          Riley 2023 instability measures.
        subgroup_auc_optimism: One dict per iteration, {group: in-bag AUC − original-data AUC},
                            restricted to each subgroup. None unless subgroup_masks was passed.
        subgroup_auc_apparent: One dict per iteration, {group: apparent-model AUC within the
                            subgroup of that resample}. None unless subgroup_masks was passed.
        apparent_params:    Best hyperparameters (C, l1_ratio) from the apparent model.
        boot_params:        Best (C, l1_ratio) from each bootstrap iteration.
        boot_coefs:         Shape (B, n_features) — each iteration's coefficients on
                            the standardized scale. Percentiles give bootstrap CIs.
        boot_scales:        Shape (B, n_features) — each iteration's Scaler SDs, so
                            boot_coefs / boot_scales un-standardizes per iteration.
        boot_spline_knots:  {source col: array (B, n_knots)} — each iteration's knot
                            locations, needed to turn its spline coefficients back
                            into a risk curve. Empty dict when no predictor is splined.
        n_iterations:       Number of bootstrap iterations completed.
        p_apparent:         Predicted probabilities from the apparent model.
        apparent_model:     Fitted _FittedModel (preprocessor + LogisticRegression) on the full dataset.
    """
    apparent_metrics: dict
    corrected_metrics: dict
    ci_lower: dict
    ci_upper: dict
    corrected_per_iter: dict
    apparent_on_resample: dict
    stability: "StabilityResult"
    apparent_params: dict
    boot_params: list[dict]
    boot_coefs: np.ndarray
    boot_scales: np.ndarray
    boot_spline_knots: dict
    n_iterations: int
    p_apparent: np.ndarray
    apparent_model: "_FittedModel"
    boot_event_counts: dict  # min/median/max events across bootstrap samples
    subgroup_auc_optimism: list[dict] | None = None  # per iteration: {group: in-bag AUC − original AUC}
    subgroup_auc_apparent: list[dict] | None = None  # per iteration: {group: apparent-model AUC on resample}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def develop_model(
    X: pd.DataFrame,
    y: np.ndarray,
    pipeline_cfg: dict,
) -> _FittedModel:
    """Fit preprocessing + GridSearchCV-tuned LogisticRegression on the given data.

    This is the single model development step reused by both the apparent
    fit (full data) and every bootstrap iteration (resampled data).

    Args:
        X: Raw feature DataFrame (uncleaned).
        y: Binary target array aligned with X.
        pipeline_cfg: Full pipeline config dict (from pipeline.yaml).

    Returns:
        _FittedModel with fitted preprocessor, best model, params, and
        a boolean non_zero_mask indicating which features have |coef| > 0.
    """
    if "grid_search" not in pipeline_cfg:
        raise KeyError(
            "Missing 'grid_search' section in pipeline config. "
            "Define C, l1_ratio, cv_folds, scoring, and n_jobs under "
            "'grid_search:' in config/pipeline.yaml."
        )
    gs_cfg = pipeline_cfg["grid_search"]
    for required in ("C", "l1_ratio", "cv_folds", "scoring", "n_jobs"):
        if required not in gs_cfg:
            raise KeyError(
                f"Missing 'grid_search.{required}' in pipeline config."
            )
    # logreg__ prefix routes param_grid into the pipeline's final step so that
    # GridSearchCV refits preprocessing inside every inner CV fold (no leakage).
    param_grid = {
        "logreg__C":        gs_cfg["C"],
        "logreg__l1_ratio": gs_cfg["l1_ratio"],
    }
    cv_folds = gs_cfg["cv_folds"]

    lr_cfg = pipeline_cfg["models"]["logistic_regression"]
    # penalty="elasticnet" removed: deprecated in sklearn 1.8 (removed in 1.10).
    # l1_ratio on the constructor is the new API; GridSearchCV overrides it
    # via param_grid so this value is never actually used for fitting.
    lr = LogisticRegression(
        solver="saga",
        l1_ratio=0.5,
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg.get("class_weight"),
        random_state=42,
    )

    preprocess_pipeline = build_preprocessing_pipeline(pipeline_cfg)
    full_pipeline = Pipeline(list(preprocess_pipeline.steps) + [("logreg", lr)])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        full_pipeline, param_grid, cv=cv,
        scoring=gs_cfg["scoring"], n_jobs=gs_cfg["n_jobs"],
    )
    gs.fit(X, y)

    best_pipeline = gs.best_estimator_
    preprocessor = best_pipeline[:-1]
    model = best_pipeline[-1]

    X_pp = preprocessor.transform(X)
    feature_names = list(X_pp.columns) if hasattr(X_pp, "columns") else \
                    [str(i) for i in range(X_pp.shape[1])]
    non_zero_mask = np.abs(model.coef_[0]) > 0
    best_params = {k.replace("logreg__", ""): v for k, v in gs.best_params_.items()}

    # Captured here rather than re-derived later: the scaler is refitted on
    # whatever data this call saw, so a bootstrap iteration's coefficients can
    # only be un-standardized with that iteration's own means and SDs.
    scaler = preprocessor.named_steps["scaler"]
    scaler = getattr(scaler, "transformer", scaler)  # unwrap progress wrapper

    # Same reason as the scaler statistics: knots sit at quantiles of whatever data
    # this call saw, so a bootstrap iteration's spline coefficients only mean
    # something against that iteration's own knots.
    transformer = preprocessor.named_steps["transformer"]
    transformer = getattr(transformer, "transformer", transformer)
    spline_knots = {k: np.asarray(v).copy()
                    for k, v in transformer.spline_knots_.items()}

    return _FittedModel(
        pipeline=best_pipeline,
        preprocessor=preprocessor,
        model=model,
        best_params=best_params,
        non_zero_mask=non_zero_mask,
        feature_names=feature_names,
        feature_means=scaler.mean_.copy(),
        feature_scales=scaler.scale_.copy(),
        spline_knots=spline_knots,
    )


def _compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Compute AUC, Brier, calibration-in-the-large, and calibration slope.

    Args:
        y: Binary outcome array.
        p: Predicted probabilities.

    Returns:
        Dict with keys: auc, brier, calibration_in_the_large,
        calibration_slope.
    """
    return {
        "auc":                      auc(y, p),
        "brier":                    brier(y, p),
        "calibration_in_the_large": calibration_in_the_large(y, p),
        "calibration_slope":        calibration_slope(y, p),
    }


def subgroup_auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """AUC restricted to the rows selected by mask.

    Args:
        y:    Binary outcome array.
        p:    Predicted probabilities aligned with y.
        mask: Boolean array selecting the subgroup members.

    Returns:
        AUC within the subgroup, or NaN if the subgroup contains fewer than
        two rows or only one outcome class (AUC undefined).
    """
    yg, pg = y[mask], p[mask]
    if len(yg) < 2 or len(np.unique(yg)) < 2:
        return float("nan")
    return auc(yg, pg)


def _bootstrap_iteration(
    seed: int,
    X: pd.DataFrame,
    y: np.ndarray,
    p_apparent: np.ndarray,
    pipeline_cfg: dict,
    metric_keys: list[str],
    subgroup_masks: dict[str, np.ndarray] | None = None,
) -> dict | None:
    """Run a single bootstrap iteration and return its metrics and predictions.

    Performs: ordinary (unstratified) bootstrap resample per Riley & Collins
    (2024, doi:10.1002/bimj.202200302) Box 1, model redevelopment, in-bag
    predictions, OOB predictions on the original dataset, metric computation
    on both, and per-metric optimism (in-bag minus OOB). The apparent model,
    held fixed, is also scored on the same resample; those scores give the
    location-shifted CI, so no separate resampling loop is needed.

    Stratification is intentionally omitted so that outcome prevalence can
    vary across bootstrap samples — suppressing that variation would
    artificially reduce calibration-in-the-large instability (Riley & Collins
    2024, Level 1 stability).

    Args:
        seed:         RNG seed for the bootstrap resample.
        X:            Original full-dataset feature DataFrame.
        y:            Original full-dataset binary target array.
        p_apparent:   Apparent-model predicted probabilities for the full dataset.
        pipeline_cfg: Full pipeline config dict.
        metric_keys:  Metric names to compute optimism for.
        subgroup_masks: Optional {group name: boolean mask over the original
                      dataset}. When given, AUC optimism and apparent-model AUC
                      are additionally computed within each subgroup.

    Returns:
        Dict with keys: optimism, apparent_on_resample, oob_metrics, oob_probs,
        non_zero_mask, coefs, feature_scales, best_params, n_events,
        subgroup_auc_optimism, subgroup_auc_apparent.
        Returns None if the bootstrap draw contains
        only one class after 10 retry attempts (practically impossible with
        real data; guard for tiny synthetic test datasets).
    """
    n = len(y)
    rng = np.random.default_rng(seed)

    for _ in range(10):
        attempt_seed = int(rng.integers(0, 2**31))
        idx = resample(np.arange(n), replace=True, n_samples=n,
                       random_state=attempt_seed)
        if len(np.unique(y[idx])) == 2:
            break
    else:
        return None

    boot = develop_model(X.iloc[idx], y[idx], pipeline_cfg)

    inbag_probs = boot.predict_proba(X.iloc[idx])
    oob_probs = boot.predict_proba(X)

    inbag_m = _compute_metrics(y[idx], inbag_probs)
    oob_m   = _compute_metrics(y, oob_probs)

    # Same optimism definition as above, but with both AUCs restricted to the
    # subgroup. The model is still developed on the whole bootstrap sample —
    # only the evaluation is stratified.
    sub_opt = sub_app = None
    if subgroup_masks:
        sub_opt = {
            name: (subgroup_auc(y[idx], inbag_probs, mask[idx])
                   - subgroup_auc(y, oob_probs, mask))
            for name, mask in subgroup_masks.items()
        }
        sub_app = {
            name: subgroup_auc(y[idx], p_apparent[idx], mask[idx])
            for name, mask in subgroup_masks.items()
        }

    return {
        "optimism":      {k: inbag_m[k] - oob_m[k] for k in metric_keys},
        "apparent_on_resample": _compute_metrics(y[idx], p_apparent[idx]),
        "oob_metrics":   oob_m,
        "oob_probs":     oob_probs,
        "non_zero_mask": boot.non_zero_mask,
        "coefs":         boot.model.coef_[0].copy(),
        "feature_scales": boot.feature_scales,
        "spline_knots":  boot.spline_knots,
        "best_params":   boot.best_params,
        "n_events":      int(y[idx].sum()),
        "subgroup_auc_optimism": sub_opt,
        "subgroup_auc_apparent": sub_app,
    }


def bootstrap_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    pipeline_cfg: dict,
    n_iterations: int = 1000,
    random_seed: int = 42,
    reporter=None,
    n_jobs: int = -1,
    subgroup_masks: dict[str, np.ndarray] | None = None,
) -> BootstrapResult:
    """Bootstrap validation with full model redevelopment per iteration.

    Args:
        X: Raw feature DataFrame (full dataset, no prior split).
        y: Binary target array aligned with X.
        pipeline_cfg: Full pipeline config dict.
        n_iterations: Number of bootstrap iterations (default 1000).
        random_seed: RNG seed for reproducibility.
        reporter: Optional ProgressReporter for live CLI display.
        n_jobs: Parallel workers for the bootstrap loop (-1 = all cores).
            Inner GridSearchCV is forced to n_jobs=1 to avoid nested
            parallelism contention when n_jobs != 1.
        subgroup_masks: Optional {group name: boolean mask over the rows of X}.
            When given, per-iteration AUC optimism is also computed within each
            subgroup and returned as BootstrapResult.subgroup_auc_optimism.

    Returns:
        BootstrapResult with optimism-corrected metrics and stability measures.
    """
    from src.metrics.stability import compute_stability

    rng = np.random.default_rng(random_seed)
    y_arr = np.asarray(y)
    n = len(y_arr)

    iter_cfg = pipeline_cfg
    if n_jobs != 1:
        iter_cfg = copy.deepcopy(pipeline_cfg)
        iter_cfg["grid_search"]["n_jobs"] = 1
        iter_cfg.setdefault("imputation", {})["n_jobs"] = 1

    if reporter is not None:
        ctx = reporter.bootstrap_progress("LogisticRegression", n_iterations)
        ctx.__enter__()
        reporter.set_phase("Apparent model [LogisticRegression]")
    else:
        print("Fitting apparent model on full dataset...")

    try:
        apparent = develop_model(X, y_arr, pipeline_cfg)
        p_apparent = apparent.predict_proba(X)
        apparent_metrics = _compute_metrics(y_arr, p_apparent)
        metric_keys = list(apparent_metrics.keys())
        seeds = [int(rng.integers(0, 2**31)) for _ in range(n_iterations)]

        if reporter is not None:
            reporter.set_phase("Bootstrap [LogisticRegression]")

        # sklearn.utils.parallel.delayed fires a spurious UserWarning when
        # get_config() returns {} (the default), because `not {}` is True even
        # though we are correctly using sklearn's own Parallel + delayed.
        results = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`sklearn.utils.parallel.delayed`",
                category=UserWarning,
            )
            for it in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(_bootstrap_iteration)(seed, X, y_arr, p_apparent, iter_cfg,
                                              metric_keys, subgroup_masks)
                for seed in seeds
            ):
                if reporter is not None:
                    reporter.advance_iter()
                results.append(it)
    finally:
        if reporter is not None:
            ctx.__exit__(None, None, None)

    valid_results = [it for it in results if it is not None]
    n_skipped = len(results) - len(valid_results)
    if n_skipped:
        import warnings as _w
        _w.warn(
            f"{n_skipped} bootstrap iteration(s) skipped: bootstrap draw "
            "contained only one class. Results are based on "
            f"{len(valid_results)} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )
    n_completed = len(valid_results)

    boot_preds = np.empty((n_completed, n), dtype=np.float32)
    optimisms: list[dict] = []
    boot_coef_masks: list[np.ndarray] = []
    boot_params: list[dict] = []
    boot_event_counts: list[int] = []
    subgroup_optimisms: list[dict] = []
    subgroup_apparent: list[dict] = []
    apparent_on_resample: dict[str, list[float]] = {k: [] for k in metric_keys}
    boot_coefs: list[np.ndarray] = []
    boot_scales: list[np.ndarray] = []
    boot_knots: dict[str, list[np.ndarray]] = {k: [] for k in apparent.spline_knots}

    for b, it in enumerate(valid_results):
        optimisms.append(it["optimism"])
        for k in metric_keys:
            apparent_on_resample[k].append(it["apparent_on_resample"][k])
        boot_preds[b] = it["oob_probs"].astype(np.float32)
        boot_coef_masks.append(it["non_zero_mask"])
        boot_coefs.append(it["coefs"])
        boot_scales.append(it["feature_scales"])
        for col in boot_knots:
            boot_knots[col].append(it["spline_knots"][col])
        boot_params.append(it["best_params"])
        boot_event_counts.append(it["n_events"])
        if it["subgroup_auc_optimism"] is not None:
            subgroup_optimisms.append(it["subgroup_auc_optimism"])
            subgroup_apparent.append(it["subgroup_auc_apparent"])

    # Cleaner keeps a fixed column list and Encoder is built with explicit
    # categories, so every iteration yields the same feature layout. Cheap guard:
    # a mismatch would silently misalign the stacked coefficient columns.
    n_feat = len(apparent.feature_names)
    if any(c.shape[0] != n_feat for c in boot_coefs):
        raise ValueError(
            "Bootstrap iterations produced differing feature counts; "
            "coefficients cannot be stacked."
        )

    mean_optimism = {
        k: float(np.mean([o[k] for o in optimisms])) for k in metric_keys
    }
    corrected = {k: apparent_metrics[k] - mean_optimism[k] for k in metric_keys}

    # Per-iteration corrected estimate: apparent - optimism_b. Kept as the raw
    # record of how much the correction itself moved from resample to resample.
    corrected_per_iter = {
        k: [apparent_metrics[k] - o[k] for o in optimisms] for k in metric_keys
    }

    # Location-shifted bootstrap CI (Noma et al. 2021, Stat Med): the percentile
    # interval of the apparent model scored on the same B resamples, shifted by
    # the mean optimism onto the corrected estimate, keeping its asymmetry. The
    # apparent interval is only an intermediate here.
    #
    # The apparent model is held fixed rather than taking each refit's in-bag
    # score: an unpenalized intercept makes in-bag calibration-in-the-large ~0 in
    # every refit, which would collapse its interval to zero width.
    #
    # It leaves out uncertainty in the optimism estimate itself, so coverage can
    # fall short of 95% at low events per predictor. A two-stage (nested)
    # bootstrap, or Harrell's (2025) ABCLOC interval, would include it.
    ci_lower = {
        k: float(np.percentile(apparent_on_resample[k], 2.5)) - mean_optimism[k]
        for k in metric_keys
    }
    ci_upper = {
        k: float(np.percentile(apparent_on_resample[k], 97.5)) - mean_optimism[k]
        for k in metric_keys
    }

    stability = compute_stability(
        p_apparent=p_apparent,
        boot_preds=boot_preds,
        boot_coef_masks=boot_coef_masks,
        feature_names=apparent.feature_names,
        boot_params=boot_params,
    )

    ec = np.array(boot_event_counts)
    return BootstrapResult(
        apparent_metrics=apparent_metrics,
        corrected_metrics=corrected,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        corrected_per_iter=corrected_per_iter,
        apparent_on_resample=apparent_on_resample,
        stability=stability,
        apparent_params=apparent.best_params,
        boot_params=boot_params,
        boot_coefs=np.vstack(boot_coefs),
        boot_scales=np.vstack(boot_scales),
        boot_spline_knots={k: np.vstack(v) for k, v in boot_knots.items()},
        n_iterations=n_completed,
        p_apparent=p_apparent,
        apparent_model=apparent,
        boot_event_counts={
            "min":    int(ec.min()),
            "median": int(np.median(ec)),
            "max":    int(ec.max()),
        },
        subgroup_auc_optimism=subgroup_optimisms or None,
        subgroup_auc_apparent=subgroup_apparent or None,
    )
