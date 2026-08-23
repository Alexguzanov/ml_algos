"""There are implementations of KNearestNeighbours for classification and regression"""

import numpy as np
import pandas as pd

from my_ml_lib.metrics import euclidean_distance, manhattan_distance


class KNNClassifier:
    """Classifier based on K Nearest Nighbours

    Parametrs
    ---------
    k : int, default=5
        Number of closed neighbours

    p: int, default=2
        Represents distance used by fit() method.
        If equals 1 -> manhattan distance
        If equals 2 -> eucledean distance
    """

    def __init__(self, k: int = 5, p: int = 2):
        self.k = k
        self._is_fitted = False
        self.p = p

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
            if self.p == 2:
                distances = euclidean_distance(self.X, x)
            elif self.p == 1:
                distances = manhattan_distance(self.X, x)
            else:
                raise ValueError(
                    "p shoud be eihter 1 or 2. '1' for manhattan distance, '2' for euclidian distance"
                )
            best_k_indicies = np.argsort(distances)[: self.k]

            targets = np.unique(self.y[best_k_indicies], return_counts=True)
            indxs = np.argmax(targets[1])
            pred = targets[0][indxs]
            predictions.append(pred)

        return np.array(predictions)
