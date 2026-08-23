"""There are implementations of KNearestNeighbours for classification and regression"""

import numpy as np
import pandas as pd

from my_ml_lib.metrics import euclidean_distance


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self._is_fitted = False

    def fit(
        self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray | pd.Series
    ) -> None:
        # validate inputs ... in progress

        self.X = np.asarray(X_train)
        self.y = np.asarray(y_train)
        self._is_fitted = True

    def predict(self, X_test: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError(
                "You shoud use .fit(X_train, y_train) first before making predictions"
            )

        predictions = []

        for x in np.asarray(X_test):
            distances = euclidean_distance(self.X, x)
            best_k_indicies = np.argsort(distances)[: self.k]

            targets = np.unique(self.y[best_k_indicies], return_counts=True)
            indxs = np.argmax(targets[1])
            pred = targets[0][indxs]
            predictions.append(pred)

        return np.array(predictions)
