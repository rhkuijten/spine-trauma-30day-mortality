import numpy as np
import pandas as pd
import pytest
from src.preprocessing.transformers import Cleaner

CFG = {
    "cleaning": {
        "sentinels": [888, 999],
        "neuro_not_examinable": {
            "source_col": "ASIA",
            "unrecorded_values": [0, 888],
            "fill_value": 5,
        },
        "binary_map": {
            "cols": ["C", "T", "L", "S"],
            "map": {"JA": 1, "NEE": 0},
        },
        "intentie_unknown": "unknown",
        "intentie_map": {
            1: "MVA", 2: "MVA", 3: "MVA", 4: "MVA", 5: "MVA", 6: "MVA",
            7: "violence", 8: "violence", 9: "violence",
            10: "fall_low", 11: "fall_high",
            14: "other", 15: "other", 18: "other",
        },
    },
    "features": {
        "continuous": ["LEEFTIJD_EXACT", "BMI", "ISS08", "NUMBER_OF_FRACTURES"],
        "binary": ["Geslacht01", "TBI_CT", "TYPE_A_INJURY", "TYPE_B_INJURY",
                   "TYPE_C_INJURY", "TYPE_F_INJURY", "C", "T", "L", "S",
                   "NEURO_NOT_EXAMINABLE"],
        "ordinal": ["COMORB", "ASIA"],
        "categorical": ["INTENTIE_cat"],
        "drop": ["SCI_yes_no", "MFI5", "N_CODE"],
    },
    "imputation": {"max_iter": 2, "random_seed": 42},
    "encoding": {
        "intentie_categories": ["fall_high", "fall_low", "MVA", "other",
                                "unknown", "violence"],
        "intentie_reference": "fall_low",
    },
}


def make_raw_df():
    """Minimal raw SPSS-style DataFrame with sentinels, JA/NEE strings, raw INTENTIE."""
    return pd.DataFrame({
        "LEEFTIJD_EXACT": [45.0, 888.0, 70.0],
        "BMI": [22.0, 999.0, 28.0],
        "ISS08": [16.0, 9.0, 25.0],
        "NUMBER_OF_FRACTURES": [2.0, 1.0, 3.0],
        "Geslacht01": [1.0, 0.0, 1.0],
        "TBI_CT": [0.0, 1.0, 0.0],
        "TYPE_A_INJURY": [1.0, 0.0, 0.0],
        "TYPE_B_INJURY": [0.0, 1.0, 0.0],
        "TYPE_C_INJURY": [0.0, 0.0, 1.0],
        "TYPE_F_INJURY": [0.0, 0.0, 0.0],
        "C": ["JA", "NEE", "JA"],
        "T": ["NEE", "JA", "NEE"],
        "L": ["NEE", "NEE", "JA"],
        "S": ["NEE", "NEE", "NEE"],
        "COMORB": [2.0, 3.0, 888.0],
        "ASIA": [4.0, 2.0, 0.0],
        "N_CODE": [1.0, 3.0, 2.0],
        "INTENTIE": [1.0, 10.0, 11.0],
        "SCI_yes_no": [0.0, 1.0, 0.0],
        "MFI5": [1.0, 2.0, 0.0],
        "D30_Mortality": [0.0, 1.0, 0.0],
    })


def test_cleaner_fit_returns_self():
    cleaner = Cleaner(CFG)
    result = cleaner.fit(make_raw_df())
    assert result is cleaner


def test_cleaner_sentinels_replaced():
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert np.isnan(out.loc[1, "LEEFTIJD_EXACT"])
    assert np.isnan(out.loc[1, "BMI"])
    assert np.isnan(out.loc[2, "COMORB"])


def test_cleaner_binary_map():
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert out.loc[0, "C"] == 1
    assert out.loc[1, "C"] == 0
    assert out.loc[2, "T"] == 0
    assert set(out["C"].dropna().unique()).issubset({0, 1})


def test_cleaner_intentie_cat_created():
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert "INTENTIE_cat" in out.columns
    assert out.loc[0, "INTENTIE_cat"] == "MVA"
    assert out.loc[1, "INTENTIE_cat"] == "fall_low"
    assert out.loc[2, "INTENTIE_cat"] == "fall_high"


def test_cleaner_raw_intentie_dropped():
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert "INTENTIE" not in out.columns


def test_cleaner_drops_columns():
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert "SCI_yes_no" not in out.columns
    assert "MFI5" not in out.columns


def test_cleaner_target_col_dropped():
    # Target column is passed as y separately — Cleaner correctly excludes it from X.
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert "D30_Mortality" not in out.columns


def test_cleaner_binary_map_numeric_values():
    df = make_raw_df().copy()
    df["C"] = [1.0, 0.0, 1.0]  # already numeric, not JA/NEE
    out = Cleaner(CFG).fit_transform(df)
    assert out.loc[0, "C"] == 1
    assert out.loc[1, "C"] == 0


def test_cleaner_unmapped_intentie_becomes_unknown():
    df = make_raw_df().copy()
    df.loc[0, "INTENTIE"] = 12.0  # code 12 not in intentie_map
    out = Cleaner(CFG).fit_transform(df)
    assert out.loc[0, "INTENTIE_cat"] == "unknown"


def test_cleaner_missing_intentie_becomes_unknown():
    df = make_raw_df().copy()
    df.loc[0, "INTENTIE"] = np.nan
    out = Cleaner(CFG).fit_transform(df)
    assert out.loc[0, "INTENTIE_cat"] == "unknown"


def test_cleaner_intentie_sentinel_becomes_unknown():
    # 888 is stripped to NaN before the mapping, so it must land in "unknown"
    # rather than reaching the imputer as a missing nominal value.
    df = make_raw_df().copy()
    df.loc[0, "INTENTIE"] = 888.0
    out = Cleaner(CFG).fit_transform(df)
    assert out.loc[0, "INTENTIE_cat"] == "unknown"


def test_cleaner_leaves_no_missing_intentie():
    df = make_raw_df().copy()
    df.loc[0, "INTENTIE"] = np.nan
    df.loc[1, "INTENTIE"] = 12.0
    out = Cleaner(CFG).fit_transform(df)
    assert out["INTENTIE_cat"].notna().all()


def test_cleaner_neuro_not_examinable_flag():
    # Row 2 has ASIA=0 ("not determinable"); rows 0 and 1 have real grades.
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert out["NEURO_NOT_EXAMINABLE"].tolist() == [0, 0, 1]


def test_cleaner_neuro_not_examinable_pins_asia_not_nan():
    # The flagged row must keep a usable ASIA value so it never reaches the imputer.
    df = make_raw_df().copy()
    df["ASIA"] = [4.0, 888.0, 0.0]
    out = Cleaner(CFG).fit_transform(df)
    assert out["NEURO_NOT_EXAMINABLE"].tolist() == [0, 1, 1]
    assert out["ASIA"].tolist() == [4.0, 5, 5]
    assert out["ASIA"].notna().all()


def test_cleaner_n_code_dropped():
    out = Cleaner(CFG).fit_transform(make_raw_df())
    assert "N_CODE" not in out.columns


def test_cleaner_drop_missing_col_no_error():
    df = make_raw_df().drop(columns=["SCI_yes_no"])  # SCI_yes_no already absent
    out = Cleaner(CFG).fit_transform(df)
    assert "SCI_yes_no" not in out.columns  # should not raise KeyError


from src.preprocessing.transformers import Imputer


def make_cleaned_df():
    """Minimal DataFrame as produced by Cleaner: numeric + INTENTIE_cat string, some NaNs."""
    return pd.DataFrame({
        "LEEFTIJD_EXACT": [45.0, np.nan, 70.0, 55.0, 63.0],
        "BMI": [22.0, 27.0, np.nan, 30.0, 24.0],
        "ISS08": [16.0, 9.0, 25.0, 12.0, 18.0],
        "NUMBER_OF_FRACTURES": [2.0, 1.0, 3.0, 2.0, 1.0],
        "Geslacht01": [1.0, 0.0, 1.0, 0.0, 1.0],
        "TBI_CT": [0.0, 1.0, 0.0, 0.0, 1.0],
        "TYPE_A_INJURY": [1.0, 0.0, 0.0, 1.0, 0.0],
        "TYPE_B_INJURY": [0.0, 1.0, 0.0, 0.0, 1.0],
        "TYPE_C_INJURY": [0.0, 0.0, 1.0, 0.0, 0.0],
        "TYPE_F_INJURY": [0.0, 0.0, 0.0, 1.0, 0.0],
        "C": [1, 0, 1, 0, 0],
        "T": [0, 1, 0, 0, 1],
        "L": [0, 0, 1, 1, 0],
        "S": [0, 0, 0, 0, 0],
        "COMORB": [2.0, 3.0, np.nan, 1.0, 4.0],
        "ASIA": [4.0, 2.0, 0.0, 5.0, 3.0],
        "N_CODE": [1.0, 3.0, 2.0, 1.0, 2.0],
        "INTENTIE_cat": ["MVA", "fall_low", "fall_high", "other", "violence"],
        "D30_Mortality": [0.0, 1.0, 0.0, 0.0, 1.0],
    })


def test_imputer_fit_returns_self():
    imp = Imputer(CFG)
    result = imp.fit(make_cleaned_df())
    assert result is imp


def test_imputer_no_nans_after_transform():
    df = make_cleaned_df()
    imp = Imputer(CFG).fit(df)
    out = imp.transform(df)
    assert out.isnull().sum().sum() == 0


def test_imputer_binary_cols_snapped():
    df = make_cleaned_df()
    imp = Imputer(CFG).fit(df)
    out = imp.transform(df)
    binary_cols = [c for c in CFG["features"]["binary"] if c in out.columns]
    for col in binary_cols:
        assert set(out[col].unique()).issubset({0, 1}), f"{col} has values outside {{0,1}}"


def test_imputer_ordinal_cols_are_integers():
    df = make_cleaned_df()
    imp = Imputer(CFG).fit(df)
    out = imp.transform(df)
    for col in CFG["features"]["ordinal"]:
        if col in out.columns:
            assert out[col].dtype in (int, np.int32, np.int64), f"{col} is not integer"


def test_imputer_columns_preserved():
    df = make_cleaned_df()
    imp = Imputer(CFG).fit(df)
    out = imp.transform(df)
    assert list(out.columns) == list(df.columns)


def test_imputer_intentie_cat_mode_filled():
    df = make_cleaned_df()
    df.loc[0, "INTENTIE_cat"] = np.nan
    imp = Imputer(CFG).fit(df)
    out = imp.transform(df)
    assert pd.notna(out.loc[0, "INTENTIE_cat"])


from src.preprocessing.transformers import Transformer


def make_imputed_df():
    """Minimal DataFrame as produced by Imputer: no NaNs, INTENTIE_cat string, numeric rest."""
    return pd.DataFrame({
        "LEEFTIJD_EXACT": [45.0, 52.0, 70.0, 55.0, 63.0],
        "BMI": [22.0, 27.0, 25.0, 30.0, 24.0],
        "ISS08": [16.0, 9.0, 25.0, 12.0, 18.0],
        "NUMBER_OF_FRACTURES": [2.0, 1.0, 3.0, 2.0, 1.0],
        "Geslacht01": [1, 0, 1, 0, 1],
        "TBI_CT": [0, 1, 0, 0, 1],
        "TYPE_A_INJURY": [1, 0, 0, 1, 0],
        "TYPE_B_INJURY": [0, 1, 0, 0, 1],
        "TYPE_C_INJURY": [0, 0, 1, 0, 0],
        "TYPE_F_INJURY": [0, 0, 0, 1, 0],
        "C": [1, 0, 1, 0, 0],
        "T": [0, 1, 0, 0, 1],
        "L": [0, 0, 1, 1, 0],
        "S": [0, 0, 0, 0, 0],
        "COMORB": [2, 3, 2, 1, 4],
        "ASIA": [4, 2, 0, 5, 3],
        "N_CODE": [1, 3, 2, 1, 2],
        "INTENTIE_cat": ["MVA", "fall_low", "fall_high", "other", "violence"],
        "D30_Mortality": [0.0, 1.0, 0.0, 0.0, 1.0],
    })


def test_transformer_fit_returns_self():
    t = Transformer(CFG)
    result = t.fit(make_imputed_df())
    assert result is t


def test_transformer_continuous_changed():
    # Transformer applies Yeo-Johnson only; standardization is Scaler's job.
    df = make_imputed_df()
    out = Transformer(CFG).fit(df).transform(df)
    for col in CFG["features"]["continuous"]:
        assert not np.allclose(df[col], out[col]), f"{col} not transformed"


def test_scaler_standardizes_every_column():
    from src.preprocessing.transformers import Scaler
    df = make_imputed_df().drop(columns=["INTENTIE_cat", "D30_Mortality"], errors="ignore")
    out = Scaler(CFG).fit(df).transform(df)
    assert list(out.columns) == list(df.columns)
    for col in out.columns:
        assert abs(out[col].mean()) < 1e-9, f"{col} mean not 0"
        if df[col].std(ddof=0) > 0:
            assert abs(out[col].std(ddof=0) - 1) < 1e-9, f"{col} sd not 1"
        else:
            # sklearn leaves zero-variance columns unscaled rather than dividing by 0
            assert out[col].eq(0).all(), f"{col} constant but not centred to 0"


def test_transformer_noncontinuous_unchanged():
    df = make_imputed_df()
    t = Transformer(CFG).fit(df)
    out = t.transform(df)
    for col in CFG["features"]["binary"] + CFG["features"]["ordinal"]:
        if col in df.columns:
            pd.testing.assert_series_equal(df[col], out[col], check_names=True)


def test_transformer_columns_preserved():
    df = make_imputed_df()
    t = Transformer(CFG).fit(df)
    out = t.transform(df)
    assert list(out.columns) == list(df.columns)


def test_transformer_no_nans_introduced():
    df = make_imputed_df()
    t = Transformer(CFG).fit(df)
    out = t.transform(df)
    assert out.isnull().sum().sum() == 0


from src.preprocessing.transformers import Encoder


def make_scaled_df():
    """Minimal DataFrame as produced by Transformer: scaled continuous, INTENTIE_cat string."""
    return pd.DataFrame({
        "LEEFTIJD_EXACT": [-0.8, 0.1, 1.3, -0.2, 0.6],
        "BMI": [-0.5, 0.4, 0.0, 1.1, -0.3],
        "ISS08": [0.3, -1.2, 1.5, -0.4, 0.5],
        "NUMBER_OF_FRACTURES": [0.2, -1.0, 1.8, 0.2, -1.0],
        "Geslacht01": [1, 0, 1, 0, 1],
        "TBI_CT": [0, 1, 0, 0, 1],
        "COMORB": [2, 3, 2, 1, 4],
        "ASIA": [4, 2, 0, 5, 3],
        "N_CODE": [1, 3, 2, 1, 2],
        "INTENTIE_cat": ["MVA", "fall_low", "fall_high", "other", "violence"],
        "D30_Mortality": [0.0, 1.0, 0.0, 0.0, 1.0],
    })


def test_encoder_fit_returns_self():
    enc = Encoder(CFG)
    result = enc.fit(make_scaled_df())
    assert result is enc


def test_encoder_creates_five_dummies():
    df = make_scaled_df()
    enc = Encoder(CFG).fit(df)
    out = enc.transform(df)
    dummy_cols = [c for c in out.columns if c.startswith("INTENTIE_")]
    assert len(dummy_cols) == 5, f"Expected 5 INTENTIE_ dummies, got {dummy_cols}"


def test_encoder_correct_dummy_names():
    df = make_scaled_df()
    enc = Encoder(CFG).fit(df)
    out = enc.transform(df)
    expected = {"INTENTIE_fall_high", "INTENTIE_MVA", "INTENTIE_other",
                "INTENTIE_unknown", "INTENTIE_violence"}
    actual = {c for c in out.columns if c.startswith("INTENTIE_")}
    assert actual == expected


def test_encoder_dummy_column_present_when_level_absent():
    # No row carries "unknown" here, but categories are declared explicitly so the
    # dummy must still appear — otherwise the feature layout would vary between
    # bootstrap resamples and coefficients could not be stacked.
    df = make_scaled_df()
    assert "unknown" not in set(df["INTENTIE_cat"])
    out = Encoder(CFG).fit(df).transform(df)
    assert "INTENTIE_unknown" in out.columns
    assert (out["INTENTIE_unknown"] == 0).all()


def test_encoder_raises_on_unlisted_category():
    # With handle_unknown="ignore" this row would silently become all-zero
    # dummies, i.e. indistinguishable from the dropped reference level.
    df = make_scaled_df().copy()
    df.loc[df.index[0], "INTENTIE_cat"] = "not_a_real_category"
    enc = Encoder(CFG).fit(make_scaled_df())
    with pytest.raises(ValueError):
        enc.transform(df)


def test_encoder_drops_intentie_cat():
    df = make_scaled_df()
    enc = Encoder(CFG).fit(df)
    out = enc.transform(df)
    assert "INTENTIE_cat" not in out.columns


def test_encoder_fall_low_is_reference():
    df = make_scaled_df()
    enc = Encoder(CFG).fit(df)
    out = enc.transform(df)
    assert "INTENTIE_fall_low" not in out.columns
    fall_low_idx = df.index[df["INTENTIE_cat"] == "fall_low"][0]
    dummy_cols = [c for c in out.columns if c.startswith("INTENTIE_")]
    assert (out.loc[fall_low_idx, dummy_cols] == 0).all()


def test_encoder_other_cols_unchanged():
    df = make_scaled_df()
    enc = Encoder(CFG).fit(df)
    out = enc.transform(df)
    for col in ["LEEFTIJD_EXACT", "Geslacht01", "COMORB", "D30_Mortality"]:
        pd.testing.assert_series_equal(df[col], out[col], check_names=True)


# ---------------------------------------------------------------------------
# Restricted cubic splines and ordinal capping
# ---------------------------------------------------------------------------

from src.preprocessing.transformers import Transformer, rcs_basis, spline_columns

SPLINE_CFG = {
    **CFG,
    "cleaning": {**CFG["cleaning"], "ordinal_caps": {"COMORB": 3}},
    "features": {
        **CFG["features"],
        "spline": {"LEEFTIJD_EXACT": {"knot_quantiles": [0.10, 0.50, 0.90]}},
    },
}


def make_age_df(n=60):
    """Continuous frame wide enough in age for quantile knots to separate."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "LEEFTIJD_EXACT": np.linspace(20.0, 90.0, n),
        "BMI": rng.uniform(18, 35, n),
        "ISS08": rng.uniform(4, 50, n),
        "NUMBER_OF_FRACTURES": rng.integers(1, 6, n).astype(float),
    })


def test_rcs_basis_shape_is_knots_minus_one():
    x = np.linspace(0, 100, 50)
    assert rcs_basis(x, [10, 50, 90]).shape == (50, 2)
    assert rcs_basis(x, [5, 35, 65, 95]).shape == (50, 3)


def test_rcs_basis_first_column_is_the_linear_term():
    x = np.linspace(20, 90, 40)
    np.testing.assert_allclose(rcs_basis(x, [25, 55, 85])[:, 0], x)


def test_rcs_basis_is_linear_beyond_the_outer_knots():
    # The restriction that gives the spline its name: outside [k0, k_last] the
    # fitted function must be a straight line, so second differences vanish there.
    knots = [30.0, 50.0, 70.0]
    x = np.array([72.0, 74.0, 76.0, 78.0])
    basis = rcs_basis(x, knots)
    for j in range(basis.shape[1]):
        second_diff = np.diff(basis[:, j], n=2)
        np.testing.assert_allclose(second_diff, 0.0, atol=1e-8)


def test_rcs_basis_rejects_too_few_knots():
    with pytest.raises(ValueError, match=">= 3 knots"):
        rcs_basis(np.arange(10.0), [10, 90])


def test_spline_columns_names():
    assert spline_columns("AGE", 2) == ["AGE_s1", "AGE_s2"]


def test_transformer_expands_spline_column_in_place():
    out = Transformer(SPLINE_CFG).fit(make_age_df()).transform(make_age_df())
    assert "LEEFTIJD_EXACT" not in out.columns
    assert list(out.columns)[:2] == ["LEEFTIJD_EXACT_s1", "LEEFTIJD_EXACT_s2"]


def test_transformer_knots_sit_at_configured_quantiles():
    df = make_age_df()
    tr = Transformer(SPLINE_CFG).fit(df)
    expected = np.quantile(df["LEEFTIJD_EXACT"], [0.10, 0.50, 0.90])
    np.testing.assert_allclose(tr.spline_knots_["LEEFTIJD_EXACT"], expected)


def test_transformer_spline_column_is_not_power_transformed():
    # Yeo-Johnson and the spline are mutually exclusive; the linear basis column
    # must still hold raw years, or the knots would not line up with it.
    df = make_age_df()
    out = Transformer(SPLINE_CFG).fit(df).transform(df)
    np.testing.assert_allclose(out["LEEFTIJD_EXACT_s1"], df["LEEFTIJD_EXACT"])


def test_transformer_still_power_transforms_the_other_continuous():
    df = make_age_df()
    out = Transformer(SPLINE_CFG).fit(df).transform(df)
    assert not np.allclose(out["ISS08"], df["ISS08"])


def test_transformer_knots_come_from_fit_data_only():
    # Knots are a fitted statistic: transforming a different frame must not move
    # them, otherwise a bootstrap iteration's coefficients would not match its knots.
    tr = Transformer(SPLINE_CFG).fit(make_age_df())
    before = tr.spline_knots_["LEEFTIJD_EXACT"].copy()
    tr.transform(make_age_df(n=25).assign(LEEFTIJD_EXACT=lambda d: d.LEEFTIJD_EXACT / 2))
    np.testing.assert_allclose(tr.spline_knots_["LEEFTIJD_EXACT"], before)


def test_transformer_expand_display_matches_transform_layout():
    df = make_age_df()
    tr = Transformer(SPLINE_CFG).fit(df)
    assert list(tr.expand_display(df).columns) == list(tr.transform(df).columns)


def test_transformer_expand_display_keeps_original_units():
    df = make_age_df()
    tr = Transformer(SPLINE_CFG).fit(df)
    disp = tr.expand_display(df)
    for col in ["LEEFTIJD_EXACT_s1", "LEEFTIJD_EXACT_s2"]:
        np.testing.assert_allclose(disp[col], df["LEEFTIJD_EXACT"])


def test_transformer_without_spline_config_is_unchanged():
    df = make_age_df()
    plain = Transformer(CFG).fit(df).transform(df)
    assert "LEEFTIJD_EXACT" in plain.columns
    assert not any(c.endswith("_s1") for c in plain.columns)


def test_cleaner_caps_ordinal_at_configured_grade():
    df = make_raw_df().copy()
    df["COMORB"] = [1.0, 4.0, 5.0]
    out = Cleaner(SPLINE_CFG).fit_transform(df)
    assert out["COMORB"].tolist() == [1.0, 3.0, 3.0]


def test_cleaner_ordinal_cap_leaves_lower_grades_alone():
    df = make_raw_df().copy()
    df["COMORB"] = [1.0, 2.0, 3.0]
    out = Cleaner(SPLINE_CFG).fit_transform(df)
    assert out["COMORB"].tolist() == [1.0, 2.0, 3.0]


def test_cleaner_without_ordinal_caps_is_unchanged():
    df = make_raw_df().copy()
    df["COMORB"] = [1.0, 4.0, 5.0]
    out = Cleaner(CFG).fit_transform(df)
    assert out["COMORB"].tolist() == [1.0, 4.0, 5.0]
