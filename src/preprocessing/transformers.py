"""Feature encoding and scaling transformers.

All transformers follow the sklearn fit/transform API so they can be composed
inside a Pipeline and fit on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell 2015, eq. 2.25): k knots -> k-1 columns.

    Column 0 is the untouched linear term and the remaining k-2 columns carry the
    curvature, so zeroing them recovers a straight line. The spline is constrained
    to be linear beyond the outer knots, which keeps the tails from swinging where
    data are sparse.

    Args:
        x:     Values to expand, shape (n,).
        knots: Ascending knot locations, at least 3.

    Returns:
        Basis matrix of shape (n, len(knots) - 1). Not standardized; Scaler is the
        last pipeline step and puts every column on a common scale.
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    n_knots = len(k)
    if n_knots < 3:
        raise ValueError(f"A restricted cubic spline needs >= 3 knots, got {n_knots}.")
    denom = (k[-1] - k[0]) ** 2
    cube = lambda v: np.where(v > 0, v ** 3, 0.0)  # noqa: E731

    cols = [x]
    for j in range(n_knots - 2):
        cols.append(
            (cube(x - k[j])
             - cube(x - k[n_knots - 2]) * (k[n_knots - 1] - k[j])
               / (k[n_knots - 1] - k[n_knots - 2])
             + cube(x - k[n_knots - 1]) * (k[n_knots - 2] - k[j])
               / (k[n_knots - 1] - k[n_knots - 2])) / denom
        )
    return np.column_stack(cols)


def spline_columns(col: str, n_basis: int) -> list[str]:
    """Names for the basis columns a spline predictor expands into.

    Args:
        col:     Source column name, e.g. "LEEFTIJD_EXACT".
        n_basis: Number of basis columns (len(knots) - 1).

    Returns:
        ["<col>_s1", ..., "<col>_s<n_basis>"], _s1 being the linear term.
    """
    return [f"{col}_s{j + 1}" for j in range(n_basis)]


class Cleaner(BaseEstimator, TransformerMixin):
    """Deterministic SPSS cleaning: sentinels → NaN, JA/NEE → 1/0, INTENTIE grouping, column drop."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def fit(self, X: pd.DataFrame, y=None) -> "Cleaner":
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cleaning = self.config["cleaning"]

        # Derived before sentinel handling: an unrecorded ASIA grade means the
        # neurological exam could not be performed, which is itself strongly
        # prognostic. Letting it become NaN would hand it to the imputer, which
        # would fill a plausible grade and erase the signal.
        neuro = cleaning["neuro_not_examinable"]
        src = neuro["source_col"]
        if src in df.columns:
            unrecorded = df[src].isin(neuro["unrecorded_values"]) | df[src].isna()
            df["NEURO_NOT_EXAMINABLE"] = unrecorded.astype(int)
            df.loc[unrecorded, src] = neuro["fill_value"]

        df = df.replace(cleaning["sentinels"], np.nan)

        # Top grades of an ordinal scale can be too sparse to estimate. Collapsing
        # them into the highest retained grade keeps the predictor ordinal and
        # monotone instead of letting a handful of patients define a whole level.
        for col, cap in cleaning.get("ordinal_caps", {}).items():
            if col in df.columns:
                df[col] = df[col].clip(upper=cap)

        for col, vals in cleaning.get("column_sentinels", {}).items():
            if col in df.columns:
                df[col] = df[col].replace(vals, np.nan)

        bmap = cleaning["binary_map"]
        full_map = {**bmap["map"], 1: 1, 0: 0, 1.0: 1, 0.0: 0}
        for col in bmap["cols"]:
            if col in df.columns:
                df[col] = df[col].map(full_map)

        # INTENTIE is nominal, so a blank or unmapped code gets its own level here
        # rather than being handed to the imputer, which would have to invent an
        # ordering over mechanisms to fill it. The encoder gives "unknown" its own
        # dummy, so the effect of an unrecorded mechanism is estimated, not assumed.
        if "INTENTIE" in df.columns:
            imap = {int(k): v for k, v in cleaning["intentie_map"].items()}
            unknown = cleaning["intentie_unknown"]
            df["INTENTIE_cat"] = df["INTENTIE"].map(
                lambda x: imap.get(int(x), unknown) if pd.notna(x) else unknown
            )
            df = df.drop(columns=["INTENTIE"])

        to_drop = [c for c in self.config["features"]["drop"] if c in df.columns]
        df = df.drop(columns=to_drop)

        feat = self.config["features"]
        keep = feat["continuous"] + feat["binary"] + feat["ordinal"] + feat["categorical"]
        df = df[[c for c in keep if c in df.columns]]

        return df


class Imputer(BaseEstimator, TransformerMixin):
    """IterativeImputer (BayesianRidge) for all columns. INTENTIE_cat is label-encoded
    before fitting and decoded after. Binary and ordinal outputs are rounded post-transform."""

    def __init__(self, config: dict) -> None:
        self.config = config
        feat = config["features"]
        self._cat_col = feat["categorical"][0]
        self._cat_to_int: dict | None = None
        self._int_to_cat: dict | None = None
        self._fitted_imputer: IterativeImputer | None = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._cat_col in df.columns and self._cat_to_int is not None:
            df[self._cat_col] = df[self._cat_col].map(self._cat_to_int)
        return df

    def _decode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._cat_col in df.columns and self._int_to_cat is not None:
            # Cleaner leaves no NaN in this column, so nothing is normally imputed
            # here. The clip guards the residual path: BayesianRidge is unbounded,
            # so an imputed code could round outside the label range, and an
            # out-of-range code would map to NaN and be silently one-hot encoded
            # as the reference category.
            codes = (df[self._cat_col].round()
                     .clip(0, len(self._int_to_cat) - 1)
                     .astype("Int64"))
            df[self._cat_col] = codes.map(self._int_to_cat)
        return df

    def fit(self, X: pd.DataFrame, y=None) -> "Imputer":
        if self._cat_col in X.columns:
            cats = sorted(X[self._cat_col].dropna().unique())
            self._cat_to_int = {c: float(i) for i, c in enumerate(cats)}
            self._int_to_cat = {i: c for i, c in enumerate(cats)}

        X_enc = self._encode(X)
        imp_cfg = self.config["imputation"]
        seed = imp_cfg["random_seed"]

        if X_enc.isnull().any().any():
            self._fitted_imputer = IterativeImputer(
                max_iter=imp_cfg["max_iter"],
                random_state=seed,
            )
            self._fitted_imputer.fit(X_enc)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = self._encode(X)

        if X_enc.isnull().any().any():
            df = pd.DataFrame(self._fitted_imputer.transform(X_enc), columns=X_enc.columns, index=X_enc.index)
        else:
            df = X_enc.copy()

        binary_cols = [c for c in self.config["features"]["binary"] if c in df.columns]
        for col in binary_cols:
            df[col] = df[col].round().clip(0, 1).astype(int)

        ordinal_cols = [c for c in self.config["features"]["ordinal"] if c in df.columns]
        for col in ordinal_cols:
            df[col] = df[col].round().astype(int)

        df = self._decode(df)
        return df[X.columns]


class Transformer(BaseEstimator, TransformerMixin):
    """Yeo-Johnson power transform for continuous predictors, restricted cubic
    splines for those whose effect on the log-odds is not linear.

    Yeo-Johnson is monotone, so it can normalise a predictor's marginal
    distribution but cannot represent a relationship that changes direction.
    Predictors listed under features.spline are therefore expanded into a spline
    basis instead of being power transformed; the two treatments are mutually
    exclusive. Knots are placed at fitted-data quantiles, so each bootstrap
    resample and each CV fold gets its own knots, as full redevelopment requires.

    Standardization is deliberately left off here and done for every predictor at
    once by Scaler, the final pipeline step, so that all columns reach the
    penalised model on a common scale.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._pt = PowerTransformer(method="yeo-johnson", standardize=False)

    def fit(self, X: pd.DataFrame, y=None) -> "Transformer":
        spline_cfg = self.config["features"].get("spline") or {}
        self._spline_knots = {
            col: np.quantile(X[col].to_numpy(dtype=float), cfg["knot_quantiles"])
            for col, cfg in spline_cfg.items() if col in X.columns
        }
        self._cont_cols = [
            c for c in self.config["features"]["continuous"]
            if c in X.columns and c not in self._spline_knots
        ]
        if self._cont_cols:
            self._pt.fit(X[self._cont_cols])
        self.is_fitted_ = True
        return self

    @property
    def spline_knots_(self) -> dict[str, np.ndarray]:
        """Fitted knot locations per spline predictor, on the pre-transform scale."""
        return self._spline_knots

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self._cont_cols:
            df[self._cont_cols] = self._pt.transform(df[self._cont_cols])
        for col, knots in self._spline_knots.items():
            basis = rcs_basis(df[col].to_numpy(dtype=float), knots)
            df = self._replace_with(df, col, spline_columns(col, basis.shape[1]),
                                    lambda j, b=basis: b[:, j])
        return df

    def expand_display(self, X: pd.DataFrame) -> pd.DataFrame:
        """Give an untransformed frame the spline layout, keeping original units.

        Each spline source column is repeated across its basis column names, so a
        display frame lines up with the model's feature layout while still holding
        readable values (age in years, not basis coordinates).

        Args:
            X: Frame carrying the spline source columns in their original units.

        Returns:
            Frame with the same columns as transform() produces.
        """
        df = X.copy()
        for col, knots in self._spline_knots.items():
            values = df[col].to_numpy(dtype=float)
            df = self._replace_with(df, col, spline_columns(col, len(knots) - 1),
                                    lambda j, v=values: v)
        return df

    @staticmethod
    def _replace_with(df, col, names, value_of):
        """Swap one column for several, in place of the original, order preserved."""
        pos = df.columns.get_loc(col)
        df = df.drop(columns=[col])
        for offset, name in enumerate(names):
            df.insert(pos + offset, name, value_of(offset))
        return df


class Scaler(BaseEstimator, TransformerMixin):
    """Standardize every predictor to mean 0, SD 1.

    Runs last, after one-hot encoding, so continuous, ordinal, binary and dummy
    columns all reach the penalised model on the same scale. Ridge penalises the
    sum of squared coefficients, so an unscaled predictor with a small spread
    would otherwise need a larger coefficient for the same effect and be shrunk
    more heavily purely because of its units.

    Column names are preserved so downstream code can map coefficients back to
    predictors.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None) -> "Scaler":
        self._cols = list(X.columns)
        self._scaler.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.transform(X[self._cols]),
            columns=self._cols,
            index=X.index,
        )

    @property
    def mean_(self) -> np.ndarray:
        """Per-column mean subtracted at transform time (fit-data statistic)."""
        return self._scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        """Per-column SD divided out at transform time (fit-data statistic).

        Exposed so a coefficient fitted on the standardized scale can be put
        back on the pre-scaling scale as beta / scale_.
        """
        return self._scaler.scale_


class Encoder(BaseEstimator, TransformerMixin):
    """One-hot encode the categorical predictor with a configurable reference level."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._cat_col = config["features"]["categorical"][0]
        enc_cfg = config["encoding"]
        # handle_unknown="error": with "ignore", an unlisted or NaN category is
        # encoded as all-zero dummies, which is indistinguishable from the dropped
        # reference level — a silent misassignment. Every code Cleaner can emit is
        # in intentie_categories, so anything else is a real data problem.
        self._ohe = OneHotEncoder(
            categories=[list(enc_cfg["intentie_categories"])],
            drop=[enc_cfg["intentie_reference"]],
            sparse_output=False,
            handle_unknown="error",
        )

    def fit(self, X: pd.DataFrame, y=None) -> "Encoder":
        self._ohe.fit(X[[self._cat_col]])
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        raw_names = self._ohe.get_feature_names_out([self._cat_col])
        prefix = self._cat_col + "_"
        dummy_cols = [n.replace(prefix, self._cat_col.replace("_cat", "_")) for n in raw_names]
        dummies = pd.DataFrame(
            self._ohe.transform(df[[self._cat_col]]),
            columns=dummy_cols,
            index=df.index,
        )
        return pd.concat([df.drop(columns=[self._cat_col]), dummies], axis=1)