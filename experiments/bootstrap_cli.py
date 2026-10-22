"""Bootstrap validation entry point for the Spine Trauma 30-day mortality model.

Runs full model development (preprocessing + GridSearchCV tuning + LogisticRegression)
on the complete dataset, then repeats N times on bootstrap resamples to
produce optimism-corrected performance estimates and Riley 2023 stability measures.

Usage:
    python -m experiments.bootstrap
    python -m experiments.bootstrap --data path/to/file.sav --iterations 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def get_git_commit() -> str:
    """Return the current HEAD commit hash (short form).

    Returns:
        7-character git commit hash, or 'unknown' if unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def print_results_table(result) -> None:
    """Print a formatted table of apparent, corrected, and CI metrics.

    Args:
        result: BootstrapResult from bootstrap_validate().
    """
    keys = list(result.apparent_metrics.keys())
    header = f"{'Metric':<18} {'Apparent':>10} {'Corrected':>10} {'95% CI':>20}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for k in keys:
        app = result.apparent_metrics[k]
        cor = result.corrected_metrics[k]
        lo  = result.ci_lower[k]
        hi  = result.ci_upper[k]
        print(f"{k:<18} {app:>10.4f} {cor:>10.4f}   ({lo:.4f} – {hi:.4f})")
    print("=" * len(header))


def _num(x, decimals: int) -> Decimal:
    """Round half-up through the decimal string; a rounded-to-zero negative loses its sign."""
    q = Decimal(str(float(x))).quantize(Decimal(1).scaleb(-decimals),
                                        rounding=ROUND_HALF_UP)
    return abs(q) if q == 0 else q


def point_cell(value, decimals: int = 2) -> str:
    """Format a point estimate alone, rounded exactly as in ci_cell()."""
    return f"{_num(value, decimals):.{decimals}f}"


def ci_cell(value, lower, upper, decimals: int = 2) -> str:
    """Format an estimate and its interval as one cell: "0.91 (0.89 to 0.93)".

    Rounding is half-up (a trailing 5 goes away from zero), not Python's default
    half-to-even, and the value is taken through its decimal string so that a
    literal 0.065 rounds on what was written rather than on the binary float
    just below it. A rounded-to-zero negative loses its sign, so a value that is
    effectively zero never prints as "-0.00".

    "to" separates the bounds rather than an en dash: calibration-in-the-large is
    negative on one side, and "-0.20--0.19" cannot be read.

    Args:
        value:    Point estimate.
        lower:    Lower interval bound.
        upper:    Upper interval bound.
        decimals: Decimal places to report.

    Returns:
        Formatted string.
    """
    return (f"{point_cell(value, decimals)} "
            f"({point_cell(lower, decimals)} to {point_cell(upper, decimals)})")


_METRIC_LABELS = {
    "auc":                      "C-statistic (AUC)",
    "brier":                    "Brier score",
    "calibration_in_the_large": "Calibration-in-the-large",
    "calibration_slope":        "Calibration slope",
}


def format_metrics_report(metrics_df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Collapse each estimate and its interval into one manuscript-ready cell.

    The apparent estimate is optimistic and carries no interval of its own; only
    the corrected estimate is reported with its 95% CI.

    Args:
        metrics_df: Numeric table with apparent and corrected values and the
                    corrected bounds.
        decimals:   Decimal places to report.

    Returns:
        Three-column DataFrame: Metric, Apparent, Optimism-corrected (95% CI).
    """
    return pd.DataFrame({
        "Metric": [_METRIC_LABELS.get(m, m) for m in metrics_df["metric"]],
        "Apparent": [point_cell(r["apparent"], decimals)
                     for _, r in metrics_df.iterrows()],
        "Optimism-corrected (95% CI)": [
            ci_cell(r["corrected"], r["corrected_ci_lower"],
                    r["corrected_ci_upper"], decimals)
            for _, r in metrics_df.iterrows()
        ],
    })


def format_subgroup_report(subgroup_df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Manuscript-ready subgroup table: one cell per estimate, interval included.

    Same rounding and separator conventions as format_metrics_report(). Event
    counts carry their prevalence at one decimal, the usual convention for a
    percentage in a baseline column, rather than the two used for the AUCs.

    Args:
        subgroup_df: Output of summarise_subgroups().
        decimals:    Decimal places for the AUC columns.

    Returns:
        DataFrame with Subgroup, n, Events, apparent AUC (no interval) and
        optimism-corrected AUC with its 95% CI.
    """
    return pd.DataFrame({
        "Subgroup": subgroup_df["subgroup"],
        "n": subgroup_df["n"].astype(int),
        "Events, n (%)": [
            f"{int(e)} ({100 * p:.1f})"
            for e, p in zip(subgroup_df["events"], subgroup_df["prevalence"])
        ],
        "Apparent AUC": [
            point_cell(r.auc_apparent, decimals) for r in subgroup_df.itertuples()
        ],
        "Optimism-corrected AUC (95% CI)": [
            ci_cell(r.auc_corrected, r.auc_corrected_ci_lower,
                    r.auc_corrected_ci_upper, decimals)
            for r in subgroup_df.itertuples()
        ],
    })


def save_results_json(result, commit_hash: str, output_path: Path, model_path: Path) -> None:
    """Serialize bootstrap results to a JSON file.

    Args:
        result:      BootstrapResult from bootstrap_validate().
        commit_hash: Short git commit hash for traceability.
        output_path: Destination JSON path.
        model_path:  Path where the model artifact was saved.
    """
    results_dict = {
        "git_commit":        commit_hash,
        "n_iterations":      result.n_iterations,
        "final_model_path":  str(model_path),
        "_notes": {
            "apparent_metrics":  "Model fit and evaluated on full dataset — optimistic, not for reporting.",
            "corrected_metrics": "Point estimate: apparent − mean(optimism). Report these.",
            "corrected_ci_*":    "Location-shifted bootstrap CI (Noma et al. 2021): 2.5th/97.5th percentiles of the apparent model's metric (model held fixed) on the same bootstrap resamples used for the optimism, shifted by the mean optimism. Covers sampling variability of the metric on this cohort; does NOT include uncertainty in the optimism correction.",
            "apparent_params":   "Best C and l1_ratio from GridSearchCV on the full dataset (apparent model).",
            "boot_C":            "Best C from GridSearchCV in each bootstrap resample, one per iteration, in completion order (not seed order). l1_ratio has a single grid value, so it is not stored.",
        },
        "apparent_metrics":   result.apparent_metrics,
        "corrected_metrics":  result.corrected_metrics,
        "corrected_ci_lower": result.ci_lower,
        "corrected_ci_upper": result.ci_upper,
        "apparent_params":    result.apparent_params,
        "boot_C":             [p["C"] for p in result.boot_params],
        "overall_mape":       result.stability.overall_mape,
    }
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)


def main(
    data_path: str | None = None,
    n_iterations: int | None = None,
    n_jobs: int = -1,
) -> None:
    """Run bootstrap validation and save results.

    Args:
        data_path:    Path to raw .sav file. Defaults to config lookup.
        n_iterations: Override for number of bootstrap iterations.
        n_jobs:       Parallel workers for bootstrap loop (-1 = all cores).
    """
    from src.data.loader import load_configs, load_data
    from src.utils.mlflow_utils import log_bootstrap_run
    from src.utils.progress import ProgressReporter
    from src.validation.bootstrap import bootstrap_validate
    from src.metrics.stability import (
        plot_calibration_instability,
        plot_dca_instability,
        plot_mape_instability,
        plot_prediction_instability,
        plot_roc_instability,
        plot_subgroup_instability,
    )
    from src.metrics.metrics import dca_optout
    # Deferred: experiments.subgroup_cli imports get_git_commit from this module.
    from experiments.subgroup_cli import (
        build_subgroup_masks,
        print_table,
        summarise_subgroups,
    )

    pipeline_cfg, mlflow_cfg = load_configs()
    commit_hash = get_git_commit()

    boot_cfg = pipeline_cfg.get("bootstrap", {})
    n_iter = n_iterations or boot_cfg.get("n_iterations", 1000)
    seed   = boot_cfg.get("random_seed", 42)

    print(f"Git commit : {commit_hash}")
    print(f"Iterations : {n_iter}")

    print("Loading data...", flush=True)
    X, y = load_data(pipeline_cfg, data_path=data_path)
    print(f"Dataset    : {len(y)} patients | mortality rate: {y.mean():.3f}")

    subgroup_masks = build_subgroup_masks(X)

    reporter = ProgressReporter()
    result = bootstrap_validate(
        X, y, pipeline_cfg,
        n_iterations=n_iter,
        random_seed=seed,
        reporter=reporter,
        n_jobs=n_jobs,
        subgroup_masks=subgroup_masks,
    )

    print_results_table(result)

    print(f"\nOverall MAPE : {result.stability.overall_mape:.4f}")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    stability_dir = output_dir / "stability"
    stability_dir.mkdir(exist_ok=True)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)

    model_path = output_dir / "final_model.joblib"
    joblib.dump(result.apparent_model, model_path)
    print(f"\nFinal model saved to: {model_path}")

    out_json = output_dir / "bootstrap_results.json"
    save_results_json(result, commit_hash, out_json, model_path)
    print(f"Results saved to: {out_json}")

    # Raw per-iteration coefficients, kept so the coefficient/OR table and its
    # percentile CIs can be rebuilt without re-running the bootstrap.
    coef_path = validation_dir / "bootstrap_coefficients.npz"
    np.savez_compressed(
        coef_path,
        coefs=result.boot_coefs,
        scales=result.boot_scales,
        feature_names=np.array(result.apparent_model.feature_names),
        # Spline coefficients are only interpretable against the knots of the
        # iteration that produced them, so those travel with them.
        **{f"knots__{col}": arr
           for col, arr in result.boot_spline_knots.items()},
    )
    print(f"Bootstrap coefficients saved to: {coef_path}")

    _report_keys = ["auc", "brier", "calibration_in_the_large", "calibration_slope"]
    metrics_df = pd.DataFrame([
        {
            "metric":    k,
            "apparent":            result.apparent_metrics[k],
            "corrected":           result.corrected_metrics[k],
            "corrected_ci_lower":  result.ci_lower[k],
            "corrected_ci_upper":  result.ci_upper[k],
        }
        for k in _report_keys
    ])
    xlsx_path = validation_dir / "metrics_summary.xlsx"
    report_df = format_metrics_report(metrics_df)
    with pd.ExcelWriter(xlsx_path) as writer:
        report_df.to_excel(writer, sheet_name="report", index=False)
        metrics_df.to_excel(writer, sheet_name="numeric", index=False)
    print(f"Metrics saved to: {xlsx_path}")

    subgroup_dir = output_dir / "subgroup-analyses"
    subgroup_dir.mkdir(exist_ok=True)
    optimism_df = pd.DataFrame(result.subgroup_auc_optimism)
    apparent_df = pd.DataFrame(result.subgroup_auc_apparent)
    subgroup_df = summarise_subgroups(y, result.p_apparent, subgroup_masks,
                                      optimism_df, apparent_df)
    print_table(subgroup_df)
    subgroup_report = format_subgroup_report(subgroup_df)
    subgroup_df.to_csv(subgroup_dir / "subgroup_auc.csv", index=False)
    subgroup_report.to_csv(subgroup_dir / "subgroup_auc_report.csv", index=False)
    with pd.ExcelWriter(subgroup_dir / "subgroup_auc.xlsx") as writer:
        subgroup_report.to_excel(writer, sheet_name="report", index=False)
        subgroup_df.to_excel(writer, sheet_name="numeric", index=False)
    optimism_df.to_csv(subgroup_dir / "subgroup_optimism_per_iteration.csv", index=False)
    apparent_df.to_csv(subgroup_dir / "subgroup_apparent_auc_per_iteration.csv", index=False)
    print(f"Subgroup results saved to: {subgroup_dir}/")

    optout_thresholds = np.linspace(0.10, 0.99, 90)
    dca_df = dca_optout(y, result.p_apparent, thresholds=optout_thresholds)
    dca_path = validation_dir / "dca_optout.csv"
    dca_df.to_csv(dca_path, index=False)
    print(f"Opt-out DCA saved to: {dca_path}")

    plot_prediction_instability(
        result.stability,
        save_path=stability_dir / "prediction_instability.png",
    )
    plot_mape_instability(
        result.stability,
        save_path=stability_dir / "mape_instability.png",
    )
    plot_calibration_instability(
        y, result.p_apparent,
        result.stability,
        save_path=stability_dir / "calibration_instability.png",
    )
    plot_dca_instability(
        y, result.p_apparent,
        result.stability,
        opt_out=True,
        save_path=stability_dir / "dca_instability.png",
    )
    plot_subgroup_instability(
        result.stability,
        X,
        save_path=stability_dir / "subgroup_instability.png",
    )
    plot_roc_instability(
        y, result.p_apparent,
        result.stability,
        corrected_auc=result.corrected_metrics.get("auc"),
        save_path=stability_dir / "roc_instability.png",
    )
    print(f"Stability plots saved to: {stability_dir}/")

    log_bootstrap_run(
        result=result,
        mlflow_cfg=mlflow_cfg,
        commit_hash=commit_hash,
        n_iterations=n_iter,
        results_json=out_json,
        stability_dir=stability_dir,
    )
    print("Results logged to MLflow.")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Spine Trauma — bootstrap validation"
    )
    parser.add_argument("--data",       default=None, help="Path to raw .sav file")
    parser.add_argument("--iterations", default=None, type=int,
                        help="Number of bootstrap iterations")
    parser.add_argument("--jobs",       default=-1,   type=int,
                        help="Parallel workers for bootstrap loop (-1 = all cores)")
    args = parser.parse_args()
    main(data_path=args.data, n_iterations=args.iterations, n_jobs=args.jobs)
