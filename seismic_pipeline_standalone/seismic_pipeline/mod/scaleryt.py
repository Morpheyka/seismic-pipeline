"""
Target-aware StandardScaler wrapper.
"""

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


class PassthroughYt(TransformerMixinYt):
    """Target-aware passthrough: leaves X and y unchanged. Use in param grids to test with/without scaling."""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, copy=None):
        return X, y

