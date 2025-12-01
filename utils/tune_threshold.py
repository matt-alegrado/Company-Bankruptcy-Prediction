import numpy as np
from sklearn.metrics import precision_recall_curve


def tune_threshold(model, X_val, y_val):
	"""
	Determine the optimal classification probability threshold that maximizes F1 score
	on a validation set for the positive class. Useful for imbalanced classification problems where
	the default threshold (0.5) may not produce the best balance between precision
	and recall.

	Parameters
	----------
	model : sklearn.base.ClassifierMixin
			A fitted probabilistic classifier that implements `predict_proba()`,
			e.g., `RandomForestClassifier`.

	X_val: pandas.DataFrame or array-like of shape (n_samples, n_features)
			Validation feature set (already reduced to SHAP-selected features) used for
			threshold tuning. Column names and order must match those used during model fitting.

	y_val : array-like of shape (n_samples,)
			True binary labels corresponding to `X_val_shap` where the positive class is encoded as 1.

	Returns
	-------
	best_thr : float
			The probability threshold between 0 and 1 that maximizes the F1 score
			for predicting the positive class on the validation set.
	"""

	y_proba = model.predict_proba(X_val)[:, 1]

	prec, rec, thresholds = precision_recall_curve(y_val, y_proba)
	f1 = 2 * prec * rec / (prec + rec + 1e-9)

	best_idx = np.argmax(f1)
	if best_idx >= len(thresholds):
		best_idx = len(thresholds) - 1

	best_thr = thresholds[best_idx]
	return best_thr
