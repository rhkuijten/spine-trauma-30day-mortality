"""Explainability entry point for the Spine Trauma 30-day mortality model.

Primary output is the coefficient / odds-ratio table with bootstrap percentile
CIs: the model is linear, so the coefficients are the complete explanation and
TRIPOD+AI requires them for others to implement the model.

SHAP is kept only where it adds something the table cannot: per-patient
waterfalls (why *this* prediction), and a beeswarm showing how much each
predictor moves predictions across the cohort rather than per patient.

Usage:
    python -m explainability.run_explainability
    python -m explainability.run_explainability --patient 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_boot_coefs(path: Path, feature_names: list[str]):
    """Load per-iteration bootstrap coefficients, if the bootstrap wrote them.

    Args:
        path: Path to bootstrap_coefficients.npz.
        feature_names: Encoded feature order expected by the fitted model.

    Returns:
        (coefs, scales, knots) where coefs and scales have shape (B, n_features)
        and knots maps each splined predictor to a (B, n_knots) array, or
        (None, None, None) when the file is absent.

    Raises:
        ValueError: If the stored feature order differs from the model's, which
            would silently misalign every CI.
    """
    if not path.exists():
        return None, None, None
    z = np.load(path, allow_pickle=False)
    stored = [str(n) for n in z["feature_names"]]
    if stored != list(feature_names):
        raise ValueError(
            f"{path} was written for a different feature layout than "
            "final_model.joblib. Re-run the bootstrap."
        )
    knots = {k[len("knots__"):]: z[k] for k in z.files if k.startswith("knots__")}
    return z["coefs"], z["scales"], knots


def waterfall_indices(p: np.ndarray, percentiles: list[float]) -> dict[str, int]:
    """Pick one patient per requested percentile of predicted risk.

    Args:
        p: Predicted probabilities for the full cohort.
        percentiles: Percentiles of predicted risk to sample, e.g. [5, 50, 95].

    Returns:
        Mapping of label ("p05_risk") -> row index into p.
    """
    order = np.argsort(p)
    return {
        f"p{int(q):02d}_risk": int(order[int(round(q / 100 * (len(p) - 1)))])
        for q in percentiles
    }


def main(patient: int | None = None) -> None:
    """Write the coefficient table, forest plot, beeswarm and waterfalls.

    Args:
        patient: Optional extra row index to draw an individual waterfall for.
    """
    from src.data.loader import load_configs, load_data
    from explainability.coefficients import (
        coefficient_table,
        native_intercept,
        plot_coefficient_forest,
        plot_spline_curve,
        spline_contrasts,
    )
    from explainability.plots import plot_waterfall
    from explainability.shap_values import compute_shap, onehot_groups, spline_groups

    pipeline_cfg, _ = load_configs()
    cfg = pipeline_cfg["explainability"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    top_n = cfg["top_n"]

    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run `python -m experiments.bootstrap_cli` first."
        )
    fitted = joblib.load(model_path)
    X, y = load_data(pipeline_cfg)
    print(f"Cohort     : {len(y)} patients | mortality rate: {y.mean():.3f}")
    print(f"Model      : {model_path} | params {fitted.best_params}")

    # --- primary: coefficients / odds ratios ------------------------------
    boot_coefs, boot_scales, boot_knots = load_boot_coefs(
        Path(cfg["boot_coefs_path"]), fitted.feature_names
    )
    if boot_coefs is None:
        print("WARNING    : no bootstrap coefficients found — table has no CIs.")
    else:
        print(f"Bootstrap  : {len(boot_coefs)} iterations for coefficient CIs")

    table = coefficient_table(
        fitted, pipeline_cfg["features"],
        boot_coefs=boot_coefs, boot_scales=boot_scales,
        display_names=cfg["display_names"],
        unit_labels=cfg.get("unit_labels"),
    )
    table.to_csv(out_dir / "coefficient_table.csv", index=False)
    table.to_excel(out_dir / "coefficient_table.xlsx", index=False)
    plot_coefficient_forest(table, save_path=out_dir / "coefficient_forest.png",
                            display_names=cfg["display_names"])

    intercept = {
        "intercept_standardized": float(fitted.model.intercept_[0]),
        "intercept_native": native_intercept(fitted),
    }
    pd.Series(intercept).to_csv(out_dir / "model_intercept.csv", header=False)
    print(f"\nIntercept  : standardized {intercept['intercept_standardized']:+.4f} | "
          f"native {intercept['intercept_native']:+.4f}")
    _cols = [c for c in ("feature", "unit", "beta_std", "or_per_sd",
                         "or_native", "or_native_lo", "or_native_hi") if c in table]
    print(table[_cols].head(top_n).to_string(index=False))

    # --- splined predictors: a risk curve, not a row in the table ----------
    for source, spec in (pipeline_cfg["features"].get("spline") or {}).items():
        label = cfg["display_names"].get(source, source)
        reference = float(spec["reference"])
        observed = X[source].dropna().to_numpy(dtype=float)
        grid = np.linspace(observed.min(), observed.max(), 200)
        knots_b = (boot_knots or {}).get(source)

        curve = spline_contrasts(fitted, source, grid, reference,
                                 boot_coefs, boot_scales, knots_b)
        plot_spline_curve(curve, label, reference,
                          save_path=out_dir / f"spline_{source}.png")

        points = spline_contrasts(fitted, source, spec["report_at"], reference,
                                  boot_coefs, boot_scales, knots_b)
        points.to_csv(out_dir / f"spline_{source}.csv", index=False)
        print(f"\n{label} — odds ratio vs {reference:g} "
              f"({'bootstrap 95%' if knots_b is not None else 'point estimate'}):")
        print(points.to_string(index=False))

    # --- supplementary: per-patient contributions ---------------------------
    # No global SHAP figure: the model is linear, so mean|SHAP| is |beta| times the
    # predictor's spread and the ranking carries nothing the table above does not.
    # Individual predictions are the part the table cannot answer, so only the
    # waterfalls are kept, with each splined predictor and the mutually exclusive
    # injury-mechanism dummies summed back into one row each.
    expl = compute_shap(fitted, X, display_names=cfg["display_names"],
                        groups={**spline_groups(fitted), **onehot_groups(fitted)})

    p = fitted.predict_proba(X)
    picks = waterfall_indices(p, cfg["waterfall_percentiles"])
    if patient is not None:
        picks[f"patient_{patient}"] = patient
    for label, idx in picks.items():
        plot_waterfall(
            expl, idx, top_n=top_n,
            save_path=out_dir / f"shap_waterfall_{label}.png",
            group_threshold_pct=cfg.get("waterfall_group_threshold_pct"),
        )

    print(f"\nOutputs written to: {out_dir}/")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Spine Trauma — explainability")
    parser.add_argument("--patient", type=int, default=None,
                        help="Extra row index to draw an individual waterfall for")
    args = parser.parse_args()
    main(patient=args.patient)
