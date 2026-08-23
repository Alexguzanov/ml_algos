"""implementation of math distances"""

import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)
