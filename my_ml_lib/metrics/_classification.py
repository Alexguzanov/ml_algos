"""metrics for classification"""

import numpy as np
import pandas as pd


def accuracy_score(
    y_pred: np.ndarray | pd.DataFrame, y_test: np.ndarray | pd.DataFrame
) -> float:

    y_pred = np.asarray(y_pred)
    y_test = np.asarray(y_test)

    if len(y_pred) != len(y_test):
        raise ValueError("lenght of arrays should be same size")

    return np.mean(y_pred == y_test)
