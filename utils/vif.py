from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def reduce_vif_ratio(X, ratio=5.0):
    X_work = X.copy()
    dropped = []

    while True:
        vif_values = [variance_inflation_factor(X_work.values, i) for i in range(X_work.shape[1])]
        vif_series = pd.Series(vif_values, index=X_work.columns)

        worst_feat = vif_series.idxmax()
        worst_vif = float(vif_series[worst_feat])
        med_vif = float(vif_series.median())

        # stop if worst is no longer disproportionately large
        if worst_vif <= med_vif * ratio or X_work.shape[1] <= 1:
            break

        dropped.append((worst_feat, worst_vif))
        print(f"Dropped '{worst_feat}' with VIF={worst_vif:.2f} (> {med_vif*ratio:.2f} = {ratio}×median).")
        X_work = X_work.drop(columns=[worst_feat])

    return X_work, pd.DataFrame(dropped, columns=["feature", "vif_dropped"])
