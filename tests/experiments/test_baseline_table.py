import numpy as np
import pandas as pd

from experiments.baseline_table_cli import build_table

CFG = {
    "cleaning": {
        "neuro_not_examinable": {"source_col": "ASIA", "unrecorded_values": [0, 888],
                                 "fill_value": 5},
    },
    "features": {
        "continuous": [],
        "binary": ["NEURO_NOT_EXAMINABLE"],
        "ordinal": ["ASIA"],
        "categorical": [],
    },
    "explainability": {
        "display_names": {"ASIA": "ASIA grade",
                          "NEURO_NOT_EXAMINABLE": "Neurological exam not possible"},
    },
    "baseline_table": {
        "outcome_labels": {0: "Survived", 1: "Died"},
        "row_order": ["ASIA", "NEURO_NOT_EXAMINABLE"],
        "variable_labels": {},
        "not_examinable_code": 0,
        "level_labels": {"ASIA": {4: "D", 5: "E", 0: "Not examinable"}},
    },
}


def make_cleaned_df():
    """Cleaner-style frame: rows 2 and 3 were not examinable, so ASIA is pinned to 5."""
    return pd.DataFrame({
        "ASIA": [4, 5, 5, 5],
        "NEURO_NOT_EXAMINABLE": [0, 0, 1, 1],
    })


def asia_level(table, level):
    return table[table["Level"] == level].iloc[0]


def test_not_examinable_patients_not_counted_as_asia_e():
    table = build_table(make_cleaned_df(), np.array([0, 0, 1, 1]), CFG)
    assert asia_level(table, "E")["Overall"] == "1 (25.0)"
    assert asia_level(table, "E")["Died"] == "0 (0.0)"


def test_not_examinable_patients_get_own_asia_level():
    table = build_table(make_cleaned_df(), np.array([0, 0, 1, 1]), CFG)
    row = asia_level(table, "Not examinable")
    assert row["Overall"] == "2 (50.0)"
    assert row["Died"] == "2 (100.0)"


def test_build_table_leaves_input_asia_pinned():
    df = make_cleaned_df()
    build_table(df, np.array([0, 0, 1, 1]), CFG)
    assert df["ASIA"].tolist() == [4, 5, 5, 5]
