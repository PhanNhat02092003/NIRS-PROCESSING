"""PLS-DA / PLS-R chemometric baselines, ported from ../MT-SMART-NIR
(scripts/baselines.py there) for benchmarks that want a classic
chemometric comparison point alongside the deep models.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize


class PLSDAClassifier:
    """PLS-Discriminant Analysis for multi-class classification.

    Fits one PLSRegression on one-hot encoded labels; predicts via argmax.
    n_components is capped at min(n_components, n_features, n_samples-1).
    The n_classes-1 cap is NOT applied: PLS-DA optimises covariance in X
    space iteratively and benefits from more components than just
    n_classes-1.
    """

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self._pls = None
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        self._n_classes = int(y.max()) + 1
        nc = min(self.n_components, X.shape[1], X.shape[0] - 1)
        nc = max(nc, 1)
        Y = label_binarize(y, classes=np.arange(self._n_classes)).astype(np.float64)
        if self._n_classes == 2:
            Y = np.hstack([1 - Y, Y])
        self._pls = PLSRegression(n_components=nc, max_iter=1000, scale=True)
        self._pls.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._pls.predict(X)
        return np.argmax(scores, axis=1)
