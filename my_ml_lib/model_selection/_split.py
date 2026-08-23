"""Implementations of splitting data"""

import numpy as np
import pandas as pd


def train_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | np.ndarray,
    test_size: float = 0.25,
    shuffle: bool = True,
    random_state: int | None = None,
):

    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = X.shape[0]
    test_size = int(n_samples * 0.25)
    indices = np.arange(n_samples)
    train_size = n_samples - test_size

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
