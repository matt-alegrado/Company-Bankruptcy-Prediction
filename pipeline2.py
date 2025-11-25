"""
Unified Bankruptcy Dataset Pipeline:

Data cleaning
Fix messy column names, easier to work with
Convert everything to numbers, models can't handle text
Drop columns with too many missing values, they're useless
Fill the rest with medians, simple and robust
Clip extreme outliers, stops crazy ratios from breaking things
Remove zero variance columns, no useful data there

Feature engineering
Liquidity index: quick ratio, current ratio, cash vs liabilities
Leverage index: debt ratio, liability/equity
Profitability index: net income/assets, margins
Efficiency index: turnover ratios minus collection days
Cashflow adequacy: cash flow vs liabilities/assets
Growth momentum: asset and net value growth
Earnings stability: persistent EPS, cash flow/share

Conclusion from results
Low liquidity means trouble paying short term obligations
High leverage means risky debt load
Weak profitability or efficiency means operations struggling
Poor cashflow adequacy means solvency risk
Negative growth means fundamentals deteriorating
Unstable earnings means fragile firm
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------
# Config
# ------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VIF_THRESHOLD = 10.0   # more reasonable cutoff
MISSING_DROP_THRESHOLD = 0.40
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

INPUT_CSV = "data.csv"   # raw input file
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 1. Load & basic cleaning
# ------------------------------
df = pd.read_csv(INPUT_CSV)

# Normalize column names
df.columns = (
    df.columns.str.strip()
              .str.replace('[^A-Za-z0-9]+','_', regex=True)
              .str.lower()
)

# Ensure target column is named 'bankrupt'
target_candidates = [c for c in df.columns if "bankrupt_" in c]
if not target_candidates:
    raise ValueError("Target column containing 'bankrupt' not found.")
target = target_candidates[0]
df.rename(columns={target: "bankrupt_"}, inplace=True)

# Convert to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Drop sparse columns
df = df.dropna(axis=1, thresh=int((1-MISSING_DROP_THRESHOLD)*len(df)))

# Impute remaining missing with median
df = df.fillna(df.median())

# Winsorize outliers
for col in df.columns.drop("bankrupt_"):
    lower, upper = df[col].quantile(WINSOR_LOWER), df[col].quantile(WINSOR_UPPER)
    df[col] = np.clip(df[col], lower, upper)

# Drop zero-variance columns
df = df.loc[:, df.std() > 0]

# ------------------------------
# 2. Feature Engineering Library
# ------------------------------
def engineer_features(df):
    # Liquidity Index
    liquidity_cols = [c for c in ["current_ratio","quick_ratio","cash_current_liability"] if c in df.columns]
    if liquidity_cols:
        df["liquidity_index"] = df[liquidity_cols].mean(axis=1)

    # Leverage Index
    leverage_cols = [c for c in ["debt_ratio_","liability_to_equity","interest_expense_ratio"] if c in df.columns]
    if leverage_cols:
        df["leverage_index"] = df[leverage_cols].mean(axis=1)

    # Profitability Index
    profit_cols = [c for c in ["net_income_to_total_assets","gross_profit_to_sales","operating_gross_margin"] if c in df.columns]
    if profit_cols:
        df["profitability_index"] = df[profit_cols].mean(axis=1)

    # Efficiency Index
    eff_cols = [c for c in ["total_asset_turnover","inventory_turnover_rate_times","average_collection_days"] if c in df.columns]
    if eff_cols:
        df["efficiency_index"] = (
            df.get("total_asset_turnover",0) + df.get("inventory_turnover_rate_times",0) - df.get("average_collection_days",0)
        )

    # Cashflow Adequacy
    cash_cols = [c for c in ["cash_flow_to_liability","cash_flow_to_assets"] if c in df.columns]
    if cash_cols:
        df["cashflow_adequacy"] = df[cash_cols].mean(axis=1)

    # Growth Momentum
    growth_cols = [c for c in ["net_value_growth_rate","total_asset_growth_rate"] if c in df.columns]
    if growth_cols:
        df["growth_momentum"] = df[growth_cols].mean(axis=1)

    # Earnings Stability
    earn_cols = [c for c in ["persistent_eps_in_the_last_four_seasons","cash_flow_per_share"] if c in df.columns]
    if earn_cols:
        df["earnings_stability"] = df[earn_cols].mean(axis=1)

    return df

df = engineer_features(df)

# ------------------------------
# 3. Theme-based feature selection
# ------------------------------
y = df["bankrupt_"]
X = df.drop(columns=["bankrupt_"])

themes = {
    "liquidity": ["liquidity_index","current_ratio","quick_ratio","cash_current_liability"],
    "leverage": ["leverage_index","debt_ratio_","liability_to_equity","total_debt_total_net_worth"],
    "profitability": ["profitability_index","net_income_to_total_assets","gross_profit_to_sales","operating_gross_margin"],
    "efficiency": ["efficiency_index","total_asset_turnover","accounts_receivable_turnover","average_collection_days"],
    "cashflow": ["cashflow_adequacy","cash_flow_rate","cash_flow_to_liability","cash_flow_to_assets"],
    "per_share": ["earnings_stability","persistent_eps_in_the_last_four_seasons","cash_flow_per_share"],
    "growth": ["growth_momentum","net_value_growth_rate","total_asset_growth_rate"]
}
for theme, cols in themes.items():
    themes[theme] = [c for c in cols if c in X.columns]

rf_for_selection = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf_for_selection.fit(X, y)
importances = pd.Series(rf_for_selection.feature_importances_, index=X.columns)

selected_features = []
selection_log = []
for theme, cols in themes.items():
    if not cols: continue
    imp_subset = importances[cols].sort_values(ascending=False)
    top_feats = imp_subset.index[:3]  # top 3 per theme
    for feat in top_feats:
        selected_features.append(feat)
        selection_log.append((theme, feat, imp_subset[feat]))

print("\nSelected representatives per theme (top 3):")
for theme, feat, score in selection_log:
    print(f"  [{theme}] -> {feat} (importance={score:.6f})")

X_theme = X[selected_features]

# ------------------------------
# 4. VIF-based iterative pruning
# ------------------------------
def reduce_vif(X, thresh=VIF_THRESHOLD):
    X_work = X.copy()
    dropped = []
    while True:
        vif_series = pd.Series(
            [variance_inflation_factor(X_work.values, i) for i in range(X_work.shape[1])],
            index=X_work.columns
        )
        worst_feat = vif_series.sort_values(ascending=False).index[0]
        worst_vif = float(vif_series[worst_feat])
        if worst_vif <= thresh or X_work.shape[1] <= 1:
            break
        dropped.append((worst_feat, worst_vif))
        X_work = X_work.drop(columns=[worst_feat])
        print(f"Dropped '{worst_feat}' with VIF={worst_vif:.2f} (> {thresh}).")
    return X_work, pd.DataFrame(dropped, columns=["feature","vif_dropped"])

X_reduced, vif_drops = reduce_vif(X_theme, thresh=VIF_THRESHOLD)

print("\nFinal feature set after VIF reduction:")
for c in X_reduced.columns:
    print("  -", c)

# ------------------------------
# 5. Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ------------------------------
# 6. Logistic Regression
# ------------------------------
logit = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
logit.fit(X_train, y_train)
y_pred_logit = logit.predict(X_test)
y_proba_logit = logit.predict_proba(X_test)[:,1]

print("\n=== Logistic Regression Metrics ===")
print("ROC-AUC:", roc_auc_score(y_test, y_proba_logit))
print("PR-AUC:", average_precision_score(y_test, y_proba_logit))
print(classification_report(y_test, y_pred_logit, digits=3))

# ------------------------------
# 7. Random Forest
# ------------------------------
rf = RandomForestClassifier(
    n_estimators=800,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Metrics ===")
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print("PR-AUC:", average_precision_score(y_test, y_proba_rf))
print(classification_report(y_test, y_pred_rf, digits=3))

rf_importances = pd.Series(rf.feature_importances_, index=X_reduced.columns).sort_values(ascending=False)
rf_importances.to_csv(os.path.join(OUTPUT_DIR, "rf_importances.csv"))

plt.figure(figsize=(8, 6))
sns.barplot(x=rf_importances.values, y=rf_importances.index, palette="viridis")
plt.title("Random Forest Feature Importances (Reduced Set)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_importances.png"), dpi=200)
plt.close()

# ------------------------------
# 8. Confusion matrix plots
# ------------------------------
def plot_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

plot_cm(y_test, y_pred_logit, "Confusion Matrix - Logistic", os.path.join(OUTPUT_DIR, "cm_logit.png"))
plot_cm(y_test, y_pred_rf, "Confusion Matrix - Random Forest", os.path.join(OUTPUT_DIR, "cm_rf.png"))

# ------------------------------
# 9. Save final reduced dataset
# ------------------------------
final_df = pd.concat([y, X_reduced], axis=1)
final_path = os.path.join(OUTPUT_DIR, "bankruptcy_reduced.csv")
final_df.to_csv(final_path, index=False)

# Save selection logs
pd.DataFrame(selection_log, columns=["theme","feature","importance"]).to_csv(
    os.path.join(OUTPUT_DIR, "theme_selection.csv"), index=False
)
vif_drops.to_csv(os.path.join(OUTPUT_DIR, "vif_drops.csv"), index=False)

print(f"\nArtifacts saved in: {OUTPUT_DIR}")
print(f"- Final reduced dataset: {final_path}")
print(f"- Theme selection: {os.path.join(OUTPUT_DIR, 'theme_selection.csv')}")
print(f"- VIF drops: {os.path.join(OUTPUT_DIR, 'vif_drops.csv')}")
print(f"- Logistic coefficients: {os.path.join(OUTPUT_DIR, 'logit_coefficients.csv')}")
print(f"- RF importances: {os.path.join(OUTPUT_DIR, 'rf_importances.csv')}")
print(f"- RF importances plot: {os.path.join(OUTPUT_DIR, 'rf_importances.png')}")
print(f"- Confusion matrices: {os.path.join(OUTPUT_DIR, 'cm_logit.png')}, {os.path.join(OUTPUT_DIR, 'cm_rf.png')}")
