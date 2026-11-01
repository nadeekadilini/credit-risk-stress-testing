# Credit Risk Modelling and Illustrative Stress Testing

An independent, **educational** modelling portfolio project built on the UCI Statlog German Credit dataset (1994). It does **not** represent banking production modelling, Australian loan portfolios, validated macroeconomic sensitivity, or IFRS 9 compliance.

## Quick start

From the project folder:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/model.py
streamlit run app.py
```

The model script downloads the dataset from UCI on first execution (internet required). Outputs saved to `outputs/` include model, test predictions, metrics JSON, validation plots and scenario results.

## Modelling design

- UCI dataset: 1,000 records, 20 features. Original categorical/numeric attributes.
- Response: `1=good`, `2=bad` in source. We create `bad_credit=1` for original class 2.
- Stratified random split: 75% training / 25% locked test. No temporal split possible because observation dates are unavailable.
- Median numeric imputation, standardisation, categorical mode imputation and one-hot encoding are learned **within the training pipeline** to prevent train/test leakage.
- Baseline: regularised logistic regression, using scikit-learn. Held-out metrics: ROC-AUC, average precision and Brier score. ROC and calibration plots.
- Scenario stress: fixed odds multipliers of 1, 1.5 and 2 applied to predicted PD. These are **assumed sensitivity shocks**, not empirically estimated effects of unemployment, rates or GDP.
- Simplified ECL = sum(PD × assumed LGD × exposure proxy); LGD set at 45% solely as an illustration. Loan credit amount is used as a **proxy** for EAD and historically recorded in Deutsche Marks (DM), not AUD. No claim of IFRS 9 compliance.

## Critical limitations

- Small, historical German dataset, not representative of modern Australian borrowers or portfolios.
- Bad credit class is a dataset label, not validated 12-month observed default. Model outputs are illustrative *risk probabilities* rather than regulatory-calibrated 12-month PDs.
- No macroeconomic time series or observation dates; scenario multipliers are not supported by causal or econometric macro-credit estimation.
- Some predictors encode sensitive attributes or potential proxies (including personal-status/sex and age). A real lender must evaluate legal use, fairness, stability and governance before any model deployment.
- The random test split is not temporal/out-of-time validation; calibration can be unstable with a small sample.
- Historical currency units and assumed EAD/LGD mean ECL amounts are educational illustrations only.

## Next iterations

1. Add statsmodels coefficient inference, explainability and a simpler, policy-conscious feature set.
2. Add cross-validation and a second model benchmark; preserve locked test set.
3. Add bootstrap confidence intervals, threshold/cost analysis and fairness review.
4. Later, use a **dated** lending dataset if seeking actual macroeconomic PD sensitivity modelling.

## Data citation

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0).
