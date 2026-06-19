"""
Target-aware StandardScaler wrapper.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from .sklearnbaseyt import TransformerMixinYt


class StandardScalerYt(TransformerMixinYt, StandardScaler):
    """StandardScaler that preserves y in transform output."""

    def __init__(self, regression=False, copy=True, with_mean=True, with_std=True):
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)
        TransformerMixinYt.__init__(self)
        self.regression = regression

    def fit(self, X, y=None, **fit_params):
        super().fit(X)
        return self

    def transform(self, X, y=None, copy=None):
        copy_value = True if copy is None else bool(copy)
        X_transformed = super().transform(X, copy=copy_value)
        return X_transformed, y


class MaxMinSampleScaler(TransformerMixinYt):
    """Scale each sample row independently with min-max normalization."""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, copy=None):
        X_array = np.array(X, dtype=float, copy=True if copy is None else bool(copy))
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        for idx, row in enumerate(X_array):
            finite = np.isfinite(row)
            if not np.any(finite):
                continue

            row_min = float(np.min(row[finite]))
            row_max = float(np.max(row[finite]))

            if row_max == row_min:
                X_array[idx, finite] = 0.5
                continue

            X_array[idx, finite] = (row[finite] - row_min) / (row_max - row_min)

        return X_array, y


class StandardSampleScaler(TransformerMixinYt):
    """Standardize each sample row independently."""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, copy=None):
        X_array = np.array(X, dtype=float, copy=True if copy is None else bool(copy))
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        for idx, row in enumerate(X_array):
            finite = np.isfinite(row)
            if not np.any(finite):
                continue

            row_mean = float(np.mean(row[finite]))
            row_std = float(np.std(row[finite]))

            if row_std == 0.0:
                X_array[idx, finite] = 0.0
                continue

            X_array[idx, finite] = (row[finite] - row_mean) / row_std

        return X_array, y


class PassthroughYt(TransformerMixinYt):
    """Target-aware passthrough: leaves X and y unchanged. Use in param grids to test with/without scaling."""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, copy=None):
        return X, y

