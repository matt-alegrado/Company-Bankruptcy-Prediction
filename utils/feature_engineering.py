import numpy as np
import pandas as pd

def augment_negative_samples(X_in, y_in, factor=0.3, noise_scale=0.05, random_state=42):
    """
    Create hard negative samples by:
        - selecting negative samples (y=0)
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

    # Get negative sample indices
    neg_idx = np.where(y_np == 0)[0]
    if len(neg_idx) == 0:
        return X_in, y_in

    n_new = int(len(neg_idx) * factor)
    if n_new <= 0:
        return X_in, y_in

    chosen = rng.choice(neg_idx, n_new, replace=True)
    X_neg = X_np[chosen]

    feature_std = X_np.std(axis=0, ddof=1)
    feature_std[feature_std == 0] = 1e-6

    noise = rng.normal(loc=0.0, scale=feature_std * noise_scale, size=X_neg.shape)

    X_aug = X_neg + noise
    y_aug = np.zeros(n_new, dtype=int)

    X_out = np.vstack([X_np, X_aug])
    y_out = np.concatenate([y_np, y_aug])

    # Convert back to DataFrame if needed
    if cols is not None:
        X_out = pd.DataFrame(X_out, columns=cols)

    return X_out, y_out
