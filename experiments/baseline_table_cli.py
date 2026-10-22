"""Baseline characteristics table for the Spine Trauma 30-day mortality cohort.

The table is built from the Cleaner output, so it describes exactly the
predictors the model uses, in the model's coding: dropped columns are gone,
JA/NEE is 1/0, MFI5 is capped at 3+, NUMBER_OF_FRACTURES at 6+, and INTENTIE is
collapsed into INTENTIE_cat. The one exception is ASIA: Cleaner pins it to E for
rows flagged NEURO_NOT_EXAMINABLE, and the table undoes that pin (see
unpin_not_examinable).

Two consequences of describing the cleaned frame rather than the raw one:
ASIA shows no missing values (an unrecorded exam is its own 'Not examinable'
level), and INTENTIE_cat shows none either (an unrecorded mechanism becomes the
'Not recorded' level). Every other predictor's
missingness is the amount the imputer actually has to fill.

Age is reported in years. The model expands it into a restricted cubic spline
basis and standardizes every predictor afterwards, but neither step changes the
patients being described, so the raw units are what belong in the table.

No significance tests are reported: TRIPOD discourages testing baseline
differences by outcome in a prediction-model cohort, where the table is
descriptive rather than a comparison of groups.

Usage:
    python -m experiments.baseline_table_cli
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_configs, load_data
from src.preprocessing.transformers import Cleaner


def strata(y: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Column masks for the table: whole cohort plus one per outcome level.

    Args:
        y:   Binary outcome array.
        cfg: Full pipeline config dict.

    Returns:
        Ordered {column label: boolean mask over the rows}.
    """
    labels = cfg["baseline_table"]["outcome_labels"]
    masks = {"Overall": np.ones(len(y), dtype=bool)}
    for code in sorted(labels):
        masks[labels[code]] = y == code
    return masks


def continuous_row(s: pd.Series, masks: dict[str, np.ndarray]) -> dict[str, str]:
    """Median [IQR] of a continuous predictor per stratum.

    Median rather than mean throughout: ISS and BMI are right-skewed, and mixing
    summary types across rows makes a table harder to read than using the robust
    one everywhere.

    Args:
        s:     Predictor values, missing entries as NaN.
        masks: Column masks from strata().

    Returns:
        {column label: "median [q1-q3]"}.
    """
    out = {}
    for label, mask in masks.items():
        v = s[mask].dropna()
        q1, q3 = np.percentile(v, [25, 75])
        out[label] = f"{np.median(v):.1f} [{q1:.1f}-{q3:.1f}]"
    return out


def count_row(s: pd.Series, code, masks: dict[str, np.ndarray]) -> dict[str, str]:
    """n (%) of patients whose value equals code, per stratum.

    Percentages use the number of non-missing values in the stratum as the
    denominator, so they describe the patients actually observed rather than
    being diluted by missingness.

    Args:
        s:     Predictor values, missing entries as NaN.
        code:  Level to count.
        masks: Column masks from strata().

    Returns:
        {column label: "n (pct)"}.
    """
    out = {}
    for label, mask in masks.items():
        v = s[mask].dropna()
        n = int((v == code).sum())
        out[label] = f"{n} ({100 * n / len(v):.1f})" if len(v) else "-"
    return out


def unpin_not_examinable(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Move patients whose neurology could not be examined out of ASIA grade E.

    Cleaner pins their ASIA grade to E so the model reads the NEURO_NOT_EXAMINABLE
    indicator instead of an invented grade. Counted as E in a descriptive table,
    though, they would pass as neurologically intact, and they are mostly patients
    who died. Here they get baseline_table.not_examinable_code instead.

    Args:
        df:  Cleaner output.
        cfg: Full pipeline config dict.

    Returns:
        Copy of df with ASIA recoded for the flagged rows.
    """
    src = cfg["cleaning"]["neuro_not_examinable"]["source_col"]
    out = df.copy()
    flagged = out["NEURO_NOT_EXAMINABLE"] == 1
    out.loc[flagged, src] = cfg["baseline_table"]["not_examinable_code"]
    return out


def missing_cell(s: pd.Series) -> str:
    """Missing count and percentage for one predictor over the whole cohort."""
    n = int(s.isna().sum())
    return "0" if n == 0 else f"{n} ({100 * n / len(s):.1f})"


def build_table(df: pd.DataFrame, y: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Assemble the baseline characteristics table from the cleaned frame.

    Args:
        df:  Cleaner output — model predictors in model coding, pre-imputation.
        y:   Binary outcome array aligned with df.
        cfg: Full pipeline config dict.

    Returns:
        Long-format table with columns Variable, Level, one column per stratum,
        and Missing. Rows follow baseline_table.row_order.

    Raises:
        ValueError: If row_order does not cover exactly the cleaned predictors.
    """
    df = unpin_not_examinable(df, cfg)
    feat = cfg["features"]
    table_cfg = cfg["baseline_table"]
    names = {**cfg["explainability"]["display_names"], **table_cfg["variable_labels"]}
    levels = table_cfg["level_labels"]
    order = table_cfg["row_order"]
    masks = strata(y, cfg)
    rows = []

    if set(order) != set(df.columns):
        raise ValueError(
            f"baseline_table.row_order omits {sorted(set(df.columns) - set(order))} "
            f"and adds unknown {sorted(set(order) - set(df.columns))}."
        )

    def add(variable: str, level: str, cells: dict[str, str], missing: str) -> None:
        rows.append({"Variable": variable, "Level": level, **cells, "Missing": missing})

    for col in order:
        if col in feat["continuous"]:
            add(f"{names[col]}, median [IQR]", "", continuous_row(df[col], masks),
                missing_cell(df[col]))
        elif col in feat["binary"]:
            add(f"{names[col]}, n (%)", "", count_row(df[col], 1, masks),
                missing_cell(df[col]))
        else:
            add(f"{names[col]}, n (%)", "", {k: "" for k in masks},
                missing_cell(df[col]))
            for code, label in levels[col].items():
                add("", label, count_row(df[col], code, masks), "")

    header = {"Variable": "Patients, n", "Level": "",
              **{k: str(int(m.sum())) for k, m in masks.items()}, "Missing": ""}
    return pd.DataFrame([header] + rows)


def print_table(table: pd.DataFrame) -> None:
    """Print the table to the console with columns aligned."""
    widths = {c: max(len(c), table[c].astype(str).map(len).max()) for c in table.columns}
    line = "  ".join(c.ljust(widths[c]) for c in table.columns)
    print("\n" + line)
    print("-" * len(line))
    for _, r in table.iterrows():
        print("  ".join(str(r[c]).ljust(widths[c]) for c in table.columns))


def main(data_path: str | None = None) -> None:
    """Build the baseline characteristics table and save it to outputs/.

    Args:
        data_path: Path to raw .sav file. Defaults to config lookup.
    """
    cfg, _ = load_configs()
    X, y = load_data(cfg, data_path=data_path)
    df = Cleaner(cfg).fit_transform(X)

    print(f"Cohort     : {len(y)} patients | 30-day mortality: {y.mean():.3f}")
    print(f"Predictors : {len(df.columns)} ({', '.join(df.columns)})")

    table = build_table(df, y, cfg)
    print_table(table)

    out_dir = Path(cfg["baseline_table"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "baseline_characteristics.csv", index=False)
    with pd.ExcelWriter(out_dir / "baseline_characteristics.xlsx") as writer:
        table.to_excel(writer, sheet_name="baseline", index=False)
    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spine Trauma — baseline characteristics table"
    )
    parser.add_argument("--data", default=None, help="Path to raw .sav file")
    main(data_path=parser.parse_args().data)
