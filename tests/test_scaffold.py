"""Smoke tests: verify the scaffold imports cleanly and configs load correctly.

Run with: pytest tests/
"""

import importlib
import yaml
from pathlib import Path


MODULES = [
    "src.preprocessing.transformers",
    "src.metrics.metrics",
    "src.validation.bootstrap",
    "src.metrics.stability",
]

CONFIG_FILES = [
    "config/pipeline.yaml",
    "config/mlflow.yaml",
]


def test_all_modules_import():
    """Every src module must import without errors."""
    for module in MODULES:
        importlib.import_module(module)


def test_all_configs_load():
    """Every config YAML must parse without errors and be non-empty."""
    for cfg_path in CONFIG_FILES:
        path = Path(cfg_path)
        assert path.exists(), f"Config missing: {cfg_path}"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data, f"Config is empty: {cfg_path}"


def test_pipeline_config_has_required_keys():
    """pipeline.yaml must define target, bootstrap, grid_search, cleaning, features."""
    with open("config/pipeline.yaml") as f:
        cfg = yaml.safe_load(f)
    for key in ("target", "bootstrap", "grid_search", "cleaning", "features"):
        assert key in cfg, f"pipeline.yaml missing key: {key}"
    assert "event_col" in cfg["target"]
    assert "continuous" in cfg["features"]
    assert "drop" in cfg["features"]
    assert "C" in cfg["grid_search"]
    assert "l1_ratio" in cfg["grid_search"]


def test_transformers_expose_four_classes():
    """transformers.py must export Cleaner, Imputer, Transformer, Encoder."""
    from src.preprocessing.transformers import Cleaner, Imputer, Transformer, Encoder
    for cls in (Cleaner, Imputer, Transformer, Encoder):
        assert hasattr(cls, "fit")
        assert hasattr(cls, "transform")

