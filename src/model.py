"""Leakage-aware baseline credit risk model on UCI Statlog German Credit.
Run from project root: python src/model.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, roc_curve
from sklearn.calibration import calibration_curve
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

# UCI target: 1=good, 2=bad. BAD credit is the event of interest.
ds = fetch_ucirepo(id=144)
X = ds.data.features.copy()
y_raw = ds.data.targets.iloc[:, 0]
y = (pd.to_numeric(y_raw) == 2).astype(int)
assert len(X) == 1000 and len(y) == 1000
assert set(pd.to_numeric(y_raw).unique()) == {1, 2}, 'Unexpected target coding'

# Optional: exclude demographic/proxy variables in sensitivity analysis later;
# baseline is research-only and must NOT be used for real credit decisions.
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

prep = ColumnTransformer([
    ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
    ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])
model = Pipeline([('preprocess', prep), ('logit', LogisticRegression(max_iter=2000, random_state=42))])

# Locked holdout split. No data-derived preparation outside the pipeline.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
metrics = {
    'train_rows': len(X_train), 'test_rows': len(X_test),
    'test_default_prevalence': float(y_test.mean()),
    'roc_auc': float(roc_auc_score(y_test, p_test)),
    'average_precision': float(average_precision_score(y_test, p_test)),
    'brier_score': float(brier_score_loss(y_test, p_test)),
}
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
joblib.dump(model, OUT / 'pd_model.joblib')

predictions = X_test.copy()
predictions['observed_bad_credit'] = y_test.to_numpy()
predictions['predicted_pd'] = p_test
predictions.to_csv(OUT / 'holdout_predictions.csv', index=False)

fpr, tpr, _ = roc_curve(y_test, p_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"Logistic regression, AUC={metrics['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], linestyle='--', label='Random ranking')
ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Holdout ROC curve')
ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'roc.png', dpi=160); plt.close(fig)

frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy='quantile')
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mean_pred, frac_pos, 'o-', label='Observed bad-credit frequency')
ax.plot([0, 1], [0, 1], '--', label='Ideal calibration')
ax.set(xlabel='Mean predicted risk', ylabel='Observed bad-credit frequency', title='Holdout calibration')
ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'calibration.png', dpi=160); plt.close(fig)

# Research demonstration only: exposure proxy is dataset credit amount in historical DM,
# NOT an AUD EAD. Fixed LGD and stress odds multipliers are assumptions, NOT estimates.
amount_col = next((c for c in X_test.columns if c.lower() == 'attribute5'), None)
if amount_col is None:
    amount_col = X_test.columns[4]  # UCI feature order: fifth attribute is credit amount
exposure = pd.to_numeric(X_test[amount_col]).to_numpy()
assert np.all(exposure > 0)
lgd = 0.45
scenarios = {'Baseline':1.0, 'Adverse':1.5, 'Severe':2.0}
rows=[]
for name, odds_multiplier in scenarios.items():
    base_odds = p_test / np.clip(1 - p_test, 1e-12, None)
    scenario_pd = (odds_multiplier * base_odds) / (1 + odds_multiplier * base_odds)
    rows.append({'scenario':name, 'odds_multiplier_assumption':odds_multiplier,
                 'mean_pd':float(np.mean(scenario_pd)),
                 'assumed_lgd':lgd,
                 'exposure_proxy_DM':float(exposure.sum()),
                 'illustrative_ecl_DM':float(np.sum(scenario_pd * lgd * exposure))})
pd.DataFrame(rows).to_csv(OUT / 'scenario_results.csv', index=False)
print('Holdout metrics:', json.dumps(metrics, indent=2))
print('Scenario results saved:', OUT / 'scenario_results.csv')
