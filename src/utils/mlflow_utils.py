"""MLflow logging helpers for experiment entry points."""

from __future__ import annotations

from pathlib import Path

import mlflow


def log_bootstrap_run(
    result,
    mlflow_cfg: dict,
    commit_hash: str,
    n_iterations: int,
    results_json: Path,
    stability_dir: Path,
    run_name: str = "bootstrap_validation",
    phase: str = "bootstrap",
) -> None:
    """Log a bootstrap validation run to MLflow.

    Logs apparent and corrected metrics, overall MAPE, and the results JSON
    plus stability plots as artifacts. Tags include the project tags from
    mlflow.yaml plus git commit and phase.

    Args:
        result:        BootstrapResult from bootstrap_validate().
        mlflow_cfg:    Parsed mlflow.yaml config dict.
        commit_hash:   Short git commit hash for the run.
        n_iterations:  Number of bootstrap iterations executed.
        results_json:  Path to the bootstrap_results.json artifact.
        stability_dir: Directory containing stability plot PNGs to log.
        run_name:      MLflow run name.
        phase:         Value for the 'phase' tag.
    """
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            **mlflow_cfg.get("tags", {}),
            "git_commit": commit_hash,
            "phase":      phase,
        })
        mlflow.log_param("n_iterations", n_iterations)
        for k, v in result.apparent_metrics.items():
            mlflow.log_metric(f"apparent_{k}", v)
        for k, v in result.corrected_metrics.items():
            mlflow.log_metric(f"corrected_{k}", v)
        mlflow.log_metric("overall_mape", result.stability.overall_mape)
        mlflow.log_artifact(str(results_json))
        for p in stability_dir.glob("*.png"):
            mlflow.log_artifact(str(p))
