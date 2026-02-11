"""
Target-aware PCA transformer.

Wraps sklearn.decomposition.PCA while preserving the target variable (y)
through fit/transform so it can be used inside PipelineYt.
"""

from sklearn.decomposition import PCA
from .sklearnbaseyt import TransformerMixinYt


class PCAYt(TransformerMixinYt, PCA):
    """Target-aware PCA that returns (X_transformed, y) from transform."""

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        PCA.__init__(
            self,
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )
        TransformerMixinYt.__init__(self)

    def fit(self, X, y=None):
        super().fit(X)
        return self

    def transform(self, X, y=None):
        X_transformed = super().transform(X)
        return X_transformed, y

    def fit_transform(self, X, y=None, **fit_params):
        X_transformed = super().fit_transform(X, **fit_params)
        return X_transformed, y

