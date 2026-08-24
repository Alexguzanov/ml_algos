"""Implementation of

LinearRegression with numirical methods of solving optimization problem with np.lstsq
LinearRegressionClosed with closed method
SGDRegression with stohastic gradient decent
LogisticRegression with gradient decent

"""

import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, tol: float = 1e-6):
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> None:

        X = np.asarray(X)

        X = np.column_stack([np.ones(X.shape[0]), X])
        coef = np.linalg.lstsq(X, y, rcond=self.tol)[0]

        self.intercept_ = coef[0]
        self.coef_ = coef[1:]

        self.is_fitted_ = True

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:

        if not self.is_fitted_:
            raise ValueError("You should fit the model first before making preditions")

        X = np.asarray(X)

        return self.intercept_ + X @ self.coef_
