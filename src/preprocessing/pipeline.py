"""Preprocessing pipeline assembly for the Spine Trauma project."""

from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.preprocessing.transformers import Cleaner, Encoder, Imputer, Scaler, Transformer


class _StepCallbackWrapper(BaseEstimator, TransformerMixin):
    """Thin sklearn-compatible wrapper that fires on_step() at fit time only.

    Calls on_step(step_name) in fit() and fit_transform() — not in transform()
    — because only fit-time work is slow enough to warrant a progress signal.
    """

    def __init__(self, step_name: str, transformer: TransformerMixin,
                 on_step: Callable[[str], None]) -> None:
        self.step_name = step_name
        self.transformer = transformer
        self.on_step = on_step

    def fit(self, X, y=None):
        self.on_step(self.step_name)
        self.transformer.fit(X, y)
        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.on_step(self.step_name)
        if hasattr(self.transformer, "fit_transform"):
            result = self.transformer.fit_transform(X, y, **fit_params)
        else:
            result = self.transformer.fit(X, y).transform(X)
        self.is_fitted_ = True
        return result

    def transform(self, X):
        return self.transformer.transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {"step_name": self.step_name, "transformer": self.transformer,
                "on_step": self.on_step}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_preprocessing_pipeline(
    cfg: dict,
    on_step: Callable[[str], None] | None = None,
) -> Pipeline:
    """Assemble five preprocessing steps into a single sklearn Pipeline.

    Steps are fitted on training data only; call pipeline.fit(X_train, y_train)
    then pipeline.transform(X_test) to avoid data leakage.

    When on_step is provided, each step fires on_step(step_name) at the start
    of fit/fit_transform. The callback is silent during transform() calls.

    Args:
        cfg: Pipeline config dict (from pipeline.yaml).
        on_step: Optional callable(step_name: str) for progress reporting.
            Receives one of: "Cleaning", "Imputing", "Transforming", "Encoding",
            "Scaling". Pass None to disable (default).

    Returns:
        Unfitted sklearn Pipeline with steps:
        cleaner -> imputer -> transformer -> encoder -> scaler

        Scaling runs last so that the one-hot dummies produced by the encoder are
        standardized along with everything else.
    """
    def _wrap(name: str, t: TransformerMixin) -> TransformerMixin:
        return _StepCallbackWrapper(name, t, on_step) if on_step is not None else t

    return Pipeline([
        ("cleaner",     _wrap("Cleaning",     Cleaner(cfg))),
        ("imputer",     _wrap("Imputing",     Imputer(cfg))),
        ("transformer", _wrap("Transforming", Transformer(cfg))),
        ("encoder",     _wrap("Encoding",     Encoder(cfg))),
        ("scaler",      _wrap("Scaling",      Scaler(cfg))),
    ])
