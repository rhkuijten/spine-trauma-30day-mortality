"""Subgroup performance analysis for the Spine Trauma 30-day mortality model.

Reports outcome prevalence and apparent / optimism-corrected AUC within sex and
age subgroups. The model is developed on the whole sample exactly as in the main
bootstrap validation; only the evaluation is stratified, so optimism per
subgroup is:

    optimism_g = mean_b [ AUC(in-bag_b, g) - AUC(original data, g) ]
    corrected_g = apparent_g - optimism_g

This requires refitting the model on every bootstrap resample, so the run costs
the same as the main bootstrap validation (~1.5 h for 2000 iterations).

Usage:
    python -m experiments.subgroup_cli
    python -m experiments.subgroup_cli --iterations 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bootstrap_cli import format_subgroup_report, get_git_commit

# Verified against the raw MAN_VROUW column: Geslacht01 == 0 maps to 'M'
# (n=1066, the majority — a spinal trauma cohort is male-predominant) and == 1 to
# 'V' (n=420). See EDA section 2g; data/schemas/patient.yaml agrees.
SEX_COL = "Geslacht01"
SEX_GROUPS = {"Male": 0, "Female": 1}

AGE_COL = "LEEFTIJD_EXACT"
AGE_BINS = [0, 65, 80, np.inf]
AGE_LABELS = ["<65", "65-80", ">=80"]


def build_subgroup_masks(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build boolean row masks for the reported subgroups.

    Args:
        X: Raw feature DataFrame from load_data().

    Returns:
        Ordered dict of {subgroup label: boolean mask over X's rows}, starting
        with an all-True 'Overall' mask used as a sanity check against the main
        bootstrap results.
    """
    masks: dict[str, np.ndarray] = {"Overall": np.ones(len(X), dtype=bool)}

    sex = X[SEX_COL].to_numpy(dtype=float)
    for label, code in SEX_GROUPS.items():
        masks[label] = sex == code

    age = X[AGE_COL].to_numpy(dtype=float)
    binned = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    for label in AGE_LABELS:
        masks[label] = np.asarray(binned == label)

    return masks


def summarise_subgroups(
    y: np.ndarray,
    p_apparent: np.ndarray,
    masks: dict[str, np.ndarray],
    optimism_df: pd.DataFrame,
    apparent_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the subgroup results table.

    Args:
        y:           Binary outcome array for the full cohort.
        p_apparent:  Apparent-model predicted probabilities for the full cohort.
        masks:       Subgroup masks from build_subgroup_masks().
        optimism_df: Per-iteration AUC optimism, one column per subgroup.
        apparent_df: Per-iteration apparent-model AUC within each subgroup of
                     the same bootstrap resamples, one column per subgroup.

    Returns:
        DataFrame with one row per subgroup: n, events, prevalence, apparent
        AUC, corrected AUC with 95% CI, mean optimism, and the number of
        bootstrap iterations in which the subgroup AUC was estimable.
    """
    from src.validation.bootstrap import subgroup_auc

    rows = []
    for label, mask in masks.items():
        opt = optimism_df[label].to_numpy(dtype=float)
        apparent = subgroup_auc(y, p_apparent, mask)
        corrected = apparent - float(np.nanmean(opt))
        # Location-shifted interval, as in bootstrap_validate().
        app_lo, app_hi = np.nanpercentile(apparent_df[label].to_numpy(dtype=float),
                                          [2.5, 97.5])
        rows.append({
            "subgroup":            label,
            "n":                   int(mask.sum()),
            "events":              int(y[mask].sum()),
            "prevalence":          float(y[mask].mean()),
            "auc_apparent":        apparent,
            "auc_corrected":       corrected,
            "auc_corrected_ci_lower": corrected - (apparent - app_lo),
            "auc_corrected_ci_upper": corrected + (app_hi - apparent),
            "mean_optimism":       float(np.nanmean(opt)),
            "n_iterations_used":   int(np.sum(~np.isnan(opt))),
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    """Print the subgroup results as a formatted console table.

    Args:
        df: Output of summarise_subgroups().
    """
    header = (f"{'Subgroup':<10} {'n':>5} {'Events':>7} {'Prev':>7} "
              f"{'Apparent AUC':>13} {'Corrected AUC (95% CI)':>26}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for _, r in df.iterrows():
        cor = f"{r.auc_corrected:.3f} ({r.auc_corrected_ci_lower:.3f}-{r.auc_corrected_ci_upper:.3f})"
        print(f"{r.subgroup:<10} {r.n:>5d} {r.events:>7d} {r.prevalence:>7.3f} "
              f"{r.auc_apparent:>13.3f} {cor:>26}")
    print("=" * len(header))


def main(
    data_path: str | None = None,
    n_iterations: int = 2000,
    n_jobs: int = -1,
) -> None:
    """Run the subgroup bootstrap analysis and save results.

    Args:
        data_path:    Path to raw .sav file. Defaults to config lookup.
        n_iterations: Number of bootstrap iterations (match the main run).
        n_jobs:       Parallel workers for the bootstrap loop (-1 = all cores).
    """
    from src.data.loader import load_configs, load_data
    from src.utils.progress import ProgressReporter
    from src.validation.bootstrap import bootstrap_validate

    pipeline_cfg, _ = load_configs()
    commit_hash = get_git_commit()
    seed = pipeline_cfg.get("bootstrap", {}).get("random_seed", 42)

    print(f"Git commit : {commit_hash}")
    print(f"Iterations : {n_iterations}")

    print("Loading data...", flush=True)
    X, y = load_data(pipeline_cfg, data_path=data_path)
    print(f"Dataset    : {len(y)} patients | mortality rate: {y.mean():.3f}")

    masks = build_subgroup_masks(X)
    for label, mask in masks.items():
        print(f"  {label:<10} n={int(mask.sum()):>5d}  events={int(y[mask].sum()):>4d}")

    result = bootstrap_validate(
        X, y, pipeline_cfg,
        n_iterations=n_iterations,
        random_seed=seed,
        reporter=ProgressReporter(),
        n_jobs=n_jobs,
        subgroup_masks=masks,
    )

    optimism_df = pd.DataFrame(result.subgroup_auc_optimism)
    apparent_df = pd.DataFrame(result.subgroup_auc_apparent)
    summary = summarise_subgroups(y, result.p_apparent, masks, optimism_df, apparent_df)
    print_table(summary)

    print(f"\nOverall corrected AUC from main bootstrap run: "
          f"{result.corrected_metrics['auc']:.4f} (should match the 'Overall' row)")

    out_dir = Path("outputs/subgroup-analyses")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = format_subgroup_report(summary)
    summary.to_csv(out_dir / "subgroup_auc.csv", index=False)
    report.to_csv(out_dir / "subgroup_auc_report.csv", index=False)
    with pd.ExcelWriter(out_dir / "subgroup_auc.xlsx") as writer:
        report.to_excel(writer, sheet_name="report", index=False)
        summary.to_excel(writer, sheet_name="numeric", index=False)
    optimism_df.to_csv(out_dir / "subgroup_optimism_per_iteration.csv", index=False)
    apparent_df.to_csv(out_dir / "subgroup_apparent_auc_per_iteration.csv", index=False)
    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spine Trauma — subgroup AUC and prevalence"
    )
    parser.add_argument("--data",       default=None, help="Path to raw .sav file")
    parser.add_argument("--iterations", default=2000, type=int,
                        help="Number of bootstrap iterations")
    parser.add_argument("--jobs",       default=-1,   type=int,
                        help="Parallel workers for bootstrap loop (-1 = all cores)")
    args = parser.parse_args()
    main(data_path=args.data, n_iterations=args.iterations, n_jobs=args.jobs)
