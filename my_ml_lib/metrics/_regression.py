"""Implementation of regression metrics"""

import numpy as np


def mean_absolute_error(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    return np.mean(np.abs(y_pred - y_test))


def mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    return np.mean((y_pred - y_test) ** 2)


def root_mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_pred, y_test))


def r2_score(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

    return 1 - ss_res / ss_tot
