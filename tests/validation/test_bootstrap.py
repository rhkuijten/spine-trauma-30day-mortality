"""Behavioral smoke tests for bootstrap_validate() and develop_model().

Uses a tiny synthetic dataset and minimal iterations to keep runtime < 30s.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.bootstrap import BootstrapResult, develop_model, bootstrap_validate

RNG = np.random.default_rng(0)
N = 60

# Synthetic feature matrix: 4 continuous, 2 binary, 1 categorical
_CATS = ["MVA", "fall_low", "fall_high", "other"]
_X = pd.DataFrame({
    "LEEFTIJD_EXACT": RNG.normal(60, 15, N),
    "BMI":            RNG.normal(25, 5, N),
    "ISS08":          RNG.normal(16, 8, N),
    "NUMBER_OF_FRACTURES": RNG.integers(1, 5, N).astype(float),
    "Geslacht01":     RNG.integers(0, 2, N).astype(float),
    "TBI_CT":         RNG.integers(0, 2, N).astype(float),
    "INTENTIE_cat":   [_CATS[i % 4] for i in range(N)],
})
_Y = np.array([0] * (N - 12) + [1] * 12)  # ~20% event rate

# Minimal pipeline and model configs
_PIPELINE_CFG = {
    "cleaning": {
        "sentinels": [888, 999],
        "neuro_not_examinable": {
            "source_col": "ASIA",
            "unrecorded_values": [0, 888],
            "fill_value": 5,
        },
        "binary_map": {"cols": [], "map": {}},
        "intentie_map": {},
    },
    "features": {
        "continuous":  ["LEEFTIJD_EXACT", "BMI", "ISS08", "NUMBER_OF_FRACTURES"],
        "binary":      ["Geslacht01", "TBI_CT"],
        "ordinal":     [],
        "categorical": ["INTENTIE_cat"],
        "drop":        [],
    },
    "imputation": {"max_iter": 2, "random_seed": 0},
    "encoding": {
        "intentie_categories": ["fall_high", "fall_low", "MVA", "other", "violence"],
        "intentie_reference": "fall_low",
    },
    "grid_search": {
        "C":        [0.1, 1.0],
        "l1_ratio": [0.5],
        "cv_folds": 2,
        "scoring":  "neg_log_loss",
        "n_jobs":   1,
    },
    "models": {
        "logistic_regression": {
            "C": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 200,
            "class_weight": None,
        }
    },
}


# ---------------------------------------------------------------------------
# develop_model tests
# ---------------------------------------------------------------------------

def test_develop_model_returns_fitted_model():
    m = develop_model(_X, _Y, _PIPELINE_CFG)
    assert hasattr(m, "pipeline")
    assert hasattr(m, "preprocessor")
    assert hasattr(m, "model")
    assert hasattr(m, "best_params")
    assert hasattr(m, "non_zero_mask")
    assert hasattr(m, "feature_names")


def test_preprocessing_is_inside_gridsearchcv():
    """GridSearchCV must wrap the full pipeline so each fold refits preprocessing."""
    from sklearn.pipeline import Pipeline as SklearnPipeline
    m = develop_model(_X, _Y, _PIPELINE_CFG)
    assert isinstance(m.pipeline, SklearnPipeline)
    step_names = [name for name, _ in m.pipeline.steps]
    assert "logreg" in step_names
    assert len(step_names) > 1          # at least one preprocessing step present
    assert m.pipeline[-1] is m.model   # model field is the last pipeline step


def test_develop_model_predict_proba_shape():
    m = develop_model(_X, _Y, _PIPELINE_CFG)
    p = m.predict_proba(_X)
    assert p.shape == (N,)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_develop_model_non_zero_mask_is_bool():
    m = develop_model(_X, _Y, _PIPELINE_CFG)
    assert m.non_zero_mask.dtype == bool
    assert len(m.non_zero_mask) == len(m.feature_names)


def test_develop_model_best_params_has_c_and_l1():
    m = develop_model(_X, _Y, _PIPELINE_CFG)
    assert "C" in m.best_params
    assert "l1_ratio" in m.best_params


# ---------------------------------------------------------------------------
# bootstrap_validate tests
# ---------------------------------------------------------------------------

N_ITER = 4


@pytest.fixture(scope="module")
def boot_result():
    return bootstrap_validate(
        _X, _Y, _PIPELINE_CFG,
        n_iterations=N_ITER,
        random_seed=7,
    )


def test_bootstrap_result_type(boot_result):
    assert isinstance(boot_result, BootstrapResult)


def test_bootstrap_n_iterations(boot_result):
    assert boot_result.n_iterations == N_ITER


def test_bootstrap_metric_keys(boot_result):
    expected = {"auc", "brier", "calibration_in_the_large", "calibration_slope"}
    assert set(boot_result.apparent_metrics.keys()) == expected
    assert set(boot_result.corrected_metrics.keys()) == expected
    assert set(boot_result.ci_lower.keys()) == expected
    assert set(boot_result.ci_upper.keys()) == expected


def test_bootstrap_auc_in_range(boot_result):
    assert 0.0 <= boot_result.apparent_metrics["auc"] <= 1.0
    assert 0.0 <= boot_result.corrected_metrics["auc"] <= 1.0


def test_bootstrap_ci_ordered(boot_result):
    for k in boot_result.ci_lower:
        lo, hi = boot_result.ci_lower[k], boot_result.ci_upper[k]
        if np.isnan(lo) or np.isnan(hi):
            continue  # degenerate fit on tiny synthetic data; skip
        assert lo <= hi, f"CI inverted for {k}"


def test_ci_is_location_shifted_from_same_resamples(boot_result):
    """CI = percentile interval of the fixed apparent model over the same B
    resamples used for the optimism, shifted by the mean optimism."""
    r = boot_result
    for k, draws in r.apparent_on_resample.items():
        assert len(draws) == r.n_iterations
        optimism = r.apparent_metrics[k] - r.corrected_metrics[k]
        assert r.ci_lower[k] == pytest.approx(np.percentile(draws, 2.5) - optimism,
                                              nan_ok=True)
        assert r.ci_upper[k] == pytest.approx(np.percentile(draws, 97.5) - optimism,
                                              nan_ok=True)


def test_subgroup_apparent_auc_per_iteration():
    """With subgroup masks, each iteration also scores the apparent model per subgroup."""
    masks = {"Overall": np.ones(N, dtype=bool), "Male": _X["Geslacht01"].to_numpy() == 0}
    r = bootstrap_validate(_X, _Y, _PIPELINE_CFG, n_iterations=3, random_seed=3,
                           subgroup_masks=masks)
    assert len(r.subgroup_auc_apparent) == r.n_iterations
    assert all(set(d) == set(masks) for d in r.subgroup_auc_apparent)
    overall = [d["Overall"] for d in r.subgroup_auc_apparent]
    assert overall == pytest.approx(r.apparent_on_resample["auc"])


def test_bootstrap_p_apparent_shape(boot_result):
    assert boot_result.p_apparent.shape == (N,)
    assert np.all(boot_result.p_apparent >= 0) and np.all(boot_result.p_apparent <= 1)


def test_bootstrap_stability_mape_nonneg(boot_result):
    assert boot_result.stability.overall_mape >= 0.0
    assert boot_result.stability.mape_per_individual.shape == (N,)


def test_bootstrap_stability_boot_preds_shape(boot_result):
    assert boot_result.stability.boot_preds.shape == (N_ITER, N)


def test_bootstrap_stability_predictor_freq(boot_result):
    freq = boot_result.stability.predictor_freq
    assert isinstance(freq, dict)
    assert len(freq) > 0
    for name, f in freq.items():
        assert 0.0 <= f <= 1.0, f"freq out of range for {name}: {f}"


def test_bootstrap_boot_params_length(boot_result):
    assert len(boot_result.boot_params) == N_ITER
    for p in boot_result.boot_params:
        assert "C" in p and "l1_ratio" in p


def test_bootstrap_apparent_params_keys(boot_result):
    assert isinstance(boot_result.apparent_params, dict)
    assert "C" in boot_result.apparent_params
    assert "l1_ratio" in boot_result.apparent_params


def test_bootstrap_apparent_params_json_serializable(boot_result):
    import json
    json.dumps(boot_result.apparent_params)  # must not raise


def test_results_json_has_per_resample_C(boot_result, tmp_path):
    import json
    from experiments.bootstrap_cli import save_results_json
    out = tmp_path / "bootstrap_results.json"
    save_results_json(boot_result, "abc1234", out, tmp_path / "model.joblib")
    saved = json.loads(out.read_text())
    assert saved["boot_C"] == [p["C"] for p in boot_result.boot_params]
    assert len(saved["boot_C"]) == N_ITER


def test_results_json_reports_no_apparent_ci(boot_result, tmp_path):
    """Only the corrected estimate carries an interval; the apparent one does not."""
    import json
    from experiments.bootstrap_cli import save_results_json
    out = tmp_path / "bootstrap_results.json"
    save_results_json(boot_result, "abc1234", out, tmp_path / "model.joblib")
    saved = json.loads(out.read_text())
    assert not any(k.startswith("apparent_ci") for k in saved)
    assert {"corrected_ci_lower", "corrected_ci_upper"} <= saved.keys()


# ---------------------------------------------------------------------------
# Fix #3: ordinary (unstratified) bootstrap tests
# ---------------------------------------------------------------------------

def test_boot_event_counts_present(boot_result):
    """BootstrapResult must carry event-count summaries for auditability."""
    ec = boot_result.boot_event_counts
    assert isinstance(ec, dict)
    assert {"min", "median", "max"} == set(ec.keys())
    assert ec["min"] <= ec["median"] <= ec["max"]


def test_boot_event_counts_vary():
    """Ordinary bootstrap must allow event count to vary across iterations.

    With stratify=y every iteration would have exactly sum(_Y) events; without
    stratification the counts must differ across a sufficient number of draws.
    """
    result = bootstrap_validate(
        _X, _Y, _PIPELINE_CFG,
        n_iterations=20,
        random_seed=99,
    )
    assert result.boot_event_counts["min"] < result.boot_event_counts["max"], (
        "All bootstrap samples had identical event counts — stratification may "
        "still be active."
    )


def test_one_class_bootstrap_guard():
    """Iterations with a one-class bootstrap draw must be skipped gracefully.

    We use a dataset with a single positive so that most bootstrap draws
    contain only class 0. The result should still be returned (possibly with
    fewer completed iterations than requested) without raising.
    """
    rng = np.random.default_rng(1)
    n_tiny = 20
    X_tiny = pd.DataFrame({
        "LEEFTIJD_EXACT": rng.normal(60, 15, n_tiny),
        "BMI":            rng.normal(25, 5, n_tiny),
        "ISS08":          rng.normal(16, 8, n_tiny),
        "NUMBER_OF_FRACTURES": rng.integers(1, 5, n_tiny).astype(float),
        "Geslacht01":     rng.integers(0, 2, n_tiny).astype(float),
        "TBI_CT":         rng.integers(0, 2, n_tiny).astype(float),
        "INTENTIE_cat":   [_CATS[i % 4] for i in range(n_tiny)],
    })
    # Only 1 positive so most bootstrap draws are all-zero.
    y_tiny = np.zeros(n_tiny, dtype=int)
    y_tiny[0] = 1

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = bootstrap_validate(
            X_tiny, y_tiny, _PIPELINE_CFG,
            n_iterations=5,
            random_seed=0,
        )
    # Either some iterations completed successfully, or a RuntimeWarning was
    # raised about skipped iterations — but no exception should propagate.
    assert isinstance(result, BootstrapResult)
    skipped_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    # If all 5 were skipped, n_iterations would be 0 and a warning issued.
    # If some succeeded, result is valid. Both outcomes are acceptable.
    assert result.n_iterations >= 0
