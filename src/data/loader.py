"""Config and raw-data loaders for the Spine Trauma pipeline.

Kept separate from experiment entry points so they can be reused and tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import yaml


def load_configs() -> tuple[dict, dict]:
    """Load pipeline.yaml and mlflow.yaml from config/.

    Returns:
        Tuple of (pipeline_cfg, mlflow_cfg).
    """
    with open("config/pipeline.yaml") as f:
        pipeline_cfg = yaml.safe_load(f)
    with open("config/mlflow.yaml") as f:
        mlflow_cfg = yaml.safe_load(f)
    return pipeline_cfg, mlflow_cfg


def load_data(
    pipeline_cfg: dict,
    data_path: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load raw SPSS file, apply inclusion filter, return X and y.

    Args:
        pipeline_cfg: Full pipeline config dict.
        data_path:    Optional override for raw data path (file or directory).

    Returns:
        Tuple of (X, y) where X is the raw feature DataFrame and y is the
        binary target array.
    """
    raw_dir = Path(data_path or pipeline_cfg["data"]["raw_dir"])
    sav_files = list(raw_dir.glob("*.sav")) if raw_dir.is_dir() else [raw_dir]
    if not sav_files:
        raise FileNotFoundError(f"No .sav file found in {raw_dir}")

    df, _ = pyreadstat.read_sav(str(sav_files[0]))

    if "INCLUSION" in df.columns:
        df = df[df["INCLUSION"] == 1].reset_index(drop=True)

    target_col = pipeline_cfg["target"]["event_col"]
    y_raw = df[target_col].to_numpy(dtype=float)

    if np.isnan(y_raw).any():
        raise ValueError(f"Target column '{target_col}' contains NaN values.")
    invalid = ~np.isin(y_raw, [0.0, 1.0])
    if invalid.any():
        raise ValueError(
            f"Target column '{target_col}' has non-binary values: {np.unique(y_raw[invalid])}"
        )
    y = y_raw.astype(int)

    drop_cols = [target_col] + (["INCLUSION"] if "INCLUSION" in df.columns else [])
    X = df.drop(columns=drop_cols)
    return X, y
