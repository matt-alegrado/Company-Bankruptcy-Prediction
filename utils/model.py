from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime

from .common import RANDOM_STATE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MODEL_DIR

def split_train_val_test(
    df,
    target_col: str,
    train_size: float | None = TRAIN_SIZE,
    val_size: float | None = VAL_SIZE,
    test_size: float | None = TEST_SIZE,
    random_state: int | None = RANDOM_STATE,
    stratify: bool = True
) -> tuple:
    """
    Split dataframe into train / validation / test sets.

    Args:
        df          : input dataframe
        target_col  : column name of the target label
        train_size  : proportion of data for training set
        val_size    : proportion of data for validation set
        test_size   : proportion of data for test set
        random_state: random seed
        stratify    : whether to stratify by target column

    Returns:
        df_train, df_val, df_test : three dataframes
    """
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    stratify_col = df[target_col] if stratify else None

    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    stratify_temp = df_temp[target_col] if stratify else None
    val_ratio = val_size / (val_size + test_size)  

    df_val, df_test = train_test_split(
        df_temp,
        train_size=val_ratio,
        test_size=1 - val_ratio,
        random_state=random_state,
        stratify=stratify_temp
    )

    return df_train, df_val, df_test

def baseline_model(df: pd.DataFrame, target_col: str = "bankrupt") -> pd.Series:
    """
    Baseline model: always predicts company is alive
    
    Args:
        df         : input dataframe
        target_col : column name of true labels
    
    Returns:
        y_pred : pd.Series of predicted labels
    """
    y_pred = pd.Series(0, index=df.index, name="baseline_pred")
    return y_pred

def augment_positive_samples(X_in, y_in, factor=0.3, noise_scale=0.05, random_state=RANDOM_STATE):
    """
    Create hard negative samples by:
        - selecting negative samples (y=1)
        - duplicating a portion (factor% of them)
        - adding Gaussian noise proportional to feature std

    Returns augmented X and y.
    """

    rng = np.random.RandomState(random_state)

    # Convert to numpy for easier manipulation
    if isinstance(X_in, pd.DataFrame):
        cols = X_in.columns
        X_np = X_in.values.astype(float)
    else:
        cols = None
        X_np = np.asarray(X_in, dtype=float)

    y_np = np.asarray(y_in, dtype=int)

    pos_idx = np.where(y_np == 1)[0]
    if len(pos_idx) == 0:
        return X_in, y_in

    n_new = int(len(pos_idx) * factor)
    if n_new <= 0:
        return X_in, y_in

    chosen = rng.choice(pos_idx, n_new, replace=True)
    X_pos = X_np[chosen]

    feature_std = X_np.std(axis=0, ddof=1)
    feature_std[feature_std == 0] = 1e-6

    noise = rng.normal(loc=0.0, scale=feature_std * noise_scale, size=X_pos.shape)

    X_aug = X_pos + noise
    y_aug = np.ones(n_new, dtype=int)

    # Concatenate original + augmented
    X_out = np.vstack([X_np, X_aug])
    y_out = np.concatenate([y_np, y_aug])

    # Convert back to DataFrame if needed
    if cols is not None:
        X_out = pd.DataFrame(X_out, columns=cols)

    return X_out, y_out

def rf_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    n_estimators: int | None = 800,
    max_depth: int | None = 6,
    class_weight: str | None = "balanced",
    random_state: int | None = RANDOM_STATE 
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Returns:
        trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    return model

def tune_threshold(model, X_val, y_val):
	"""
	Determine the optimal classification probability threshold that maximizes F1 score
	on a validation set for the positive class. Useful for imbalanced classification problems where
	the default threshold (0.5) may not produce the best balance between precision
	and recall.
	"""

	y_proba = model.predict_proba(X_val)[:, 1]

	prec, rec, thresholds = precision_recall_curve(y_val, y_proba)
	f1 = 2 * prec * rec / (prec + rec + 1e-9)

	best_idx = np.argmax(f1)
	if best_idx >= len(thresholds):
		best_idx = len(thresholds) - 1

	best_thr = thresholds[best_idx]
	return best_thr

def decision_tree_model(
    X_train_shap: pd.DataFrame,
    rf_model: RandomForestClassifier,
    best_thr: float,
    max_depth: int = 3,
    min_samples_leaf: int = 30,
    min_impurity_decrease: float = 0.01,
    random_state: int = RANDOM_STATE
) -> DecisionTreeClassifier:
    """
    Train a shallow Decision Tree to mimic the Random Forest predictions.
    """
    y_proxy = (rf_model.predict_proba(X_train_shap)[:, 1] > best_thr)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        random_state=random_state
    )
    model.fit(X_train_shap, y_proxy)

    return model

def save_result(
    model,
    X_train_shap: pd.DataFrame,
    y_train: pd.Series,
    best_thr: float | None,
    model_dir: str | None = MODEL_DIR,
    market_value_equity_to_total_debt_thr = 0.901,
    retained_earnings_to_total_assets_thr = 0.021
) -> None:
    """
    Save final model, feature columns, metadata, and a sample input file.
    """
    # model
    model_path = MODEL_DIR / "final_model.pkl"
    joblib.dump(model, model_path)
    print(f"Final model saved → {model_path}")

    # feature columns
    feature_columns = X_train_shap.columns.tolist()
    feat_path = MODEL_DIR / "feature_columns.pkl"
    joblib.dump(feature_columns, feat_path)
    print(f"Feature columns ({len(feature_columns)} features) saved → {feat_path}")

    # metadata
    metadata = {
        "best_threshold": best_thr,
        "model_date": datetime.now().strftime("%Y-%m-%d"),
        "market_value_equity_to_total_debt_thr": market_value_equity_to_total_debt_thr,
        "retained_earnings_to_total_assets_thr": retained_earnings_to_total_assets_thr
    }
    meta_path = MODEL_DIR / "model_metadata.pkl"
    joblib.dump(metadata, meta_path)
    print(f"Metadata & thresholds saved → {meta_path}")

    # sample input
    sample_df = X_train_shap.head(10).copy()
    sample_df["true_label"] = y_train.head(10).values
    sample_path = MODEL_DIR / "sample_input_for_testing.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Sample input saved → {sample_path}")

