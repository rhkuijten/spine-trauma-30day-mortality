# 30-day mortality after spinal trauma: prediction model code

Analysis code and aggregated results for a clinical prediction model of 30-day mortality
in adults with spinal trauma. The model is a ridge-penalised logistic regression,
internally evaluated with Harrell's bootstrap optimism correction.

Patient data are **not** included (see [Data availability](#data-availability)).

## What the analysis does

1. **Preprocessing.** Registry sentinel codes (888/999) are set to missing. A patient
   whose neurological examination could not be performed gets an explicit indicator
   instead of an imputed ASIA grade. Frailty (mFI-5) is capped at 3+ and fracture count
   at 6+, and injury mechanism is grouped, with "not recorded" as its own level. Missing
   values are imputed with `IterativeImputer`. BMI and ISS are Yeo–Johnson transformed,
   age enters as a restricted cubic spline (knots at the 10th, 50th and 90th
   percentiles), and all predictors are standardised. Settings live in
   [`config/pipeline.yaml`](config/pipeline.yaml).
2. **Model development.** Ridge (L2) logistic regression. The penalty is chosen from 21
   log-spaced values (0.001–100) by stratified 5-fold cross-validation on log loss. The
   full preprocessing pipeline is refitted inside every fold.
3. **Internal evaluation.** 1,000 bootstrap resamples. In each one the whole
   development procedure is repeated (preprocessing, tuning and final fit), and
   optimism is estimated as in-resample performance minus performance on the original
   cohort. Reported measures are the AUC, Brier score, calibration-in-the-large and
   calibration slope. Each optimism-corrected measure has a 95% location-shifted
   bootstrap confidence interval (Noma et al., *Stat Med* 2021), computed from the same
   1,000 resamples. Discrimination is also reported by sex and age group (<65, 65–<80,
   ≥80 years).
4. **Stability and clinical utility.** Prediction, calibration and decision-curve
   instability across the bootstrap models (Riley & Collins, *Biom J* 2024), plus an
   opt-out decision curve analysis.
5. **Model explanation.** Odds ratios with 95% bootstrap percentile intervals, an
   odds-ratio curve for age relative to age 50, and SHAP values for individual
   predictions.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/00_eda.ipynb` | Exploratory data analysis: cohort definition, variable choices, missingness, data checks, distributions. Outputs are cleared because some tables contain counts of fewer than five patients; the code and text document each decision |
| `notebooks/01_sample_size.ipynb` | Minimum sample size for model development (Riley et al. criteria) |
| `src/preprocessing/` | Cleaning, imputation, transformation and encoding steps, and the pipeline builder |
| `src/validation/bootstrap.py` | Model development and bootstrap optimism correction with confidence intervals |
| `src/metrics/` | Performance measures, decision curves and stability measures |
| `src/data/loader.py` | Reads the registry export and applies the inclusion filter |
| `experiments/` | Command-line entry points (see below) |
| `explainability/` | Coefficient/odds-ratio table, age curve and SHAP plots |
| `config/` | All analysis settings |
| `data/schemas/` | Expected coding of the model's variables and their value labels (no data) |
| `tests/` | Unit tests on synthetic data |
| `outputs/` | Results of the final run (below) |

## Results in `outputs/`

| File | Contents |
|---|---|
| `bootstrap_results.json` | Apparent and optimism-corrected performance with 95% CIs, and the selected penalty |
| `validation/metrics_summary.xlsx` | The same performance measures as a formatted table |
| `validation/dca_optout.csv` | Opt-out decision curve (standardised net benefit) |
| `validation/bootstrap_coefficients.npz` | Coefficients of all 1,000 bootstrap models |
| `subgroup-analyses/` | AUC by sex and age group, with per-resample values |
| `stability/` | Prediction, MAPE, calibration, ROC, decision-curve and subgroup instability plots |
| `explainability/` | Odds-ratio table and forest plot, age odds-ratio curve, model intercept |
| `baseline/` | Baseline characteristics by 30-day outcome |
| `final_model.joblib` | The fitted model: preprocessing pipeline plus logistic regression |

## Running the analysis

Developed with Python 3.14 on Windows.

```bash
pip install -r requirements.txt
```

Place the registry export (SPSS `.sav`) in `data/raw/`, then run from the repository root:

```bash
python -m experiments.baseline_table_cli        # baseline characteristics
python -m experiments.bootstrap_cli             # model, bootstrap evaluation, subgroups, stability (~1 h)
python -m explainability.run_explainability     # odds ratios, age curve, SHAP plots
pytest                                          # unit tests (no data needed)
```

The notebooks are run from the `notebooks/` folder with Jupyter.

## Using the fitted model

```python
import joblib

model = joblib.load("outputs/final_model.joblib")  # run from the repository root
risk = model.predict_proba(X)                      # predicted 30-day mortality
```

`X` is a pandas DataFrame with the raw predictor columns, coded as described in
`data/schemas/patient.yaml`. Preprocessing and imputation are part of the model.
Loading the file needs this repository's `src/` package and the scikit-learn version in
`requirements.txt`. As with any pickle, only load model files from a source you trust.

## Data availability

The patient data used in this study contain personal health information. They are not
included in this repository and are not publicly available. Questions about data access
can be directed to the corresponding author of the accompanying article.

## Citation

The accompanying article is in preparation. A citation will be added here once it is
published.

## License

[MIT](LICENSE)
