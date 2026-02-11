
"""
Base classes for target-aware transformers.

This module provides the fundamental base classes for all target-aware transformers
in the mod package, ensuring consistent behavior and interface across all transformers.
"""

import logging
from sklearn.base import BaseEstimator, TransformerMixin
from ..seismo.logging_config import get_mod_logger


class TransformerMixinYt(TransformerMixin, BaseEstimator):
    """
    Mixin class for all target-aware transformers in scikit-learn.
    
    This class provides the base functionality for all transformers in the mod package,
    ensuring that target variables (y) are preserved and transformed alongside feature
    data (X) throughout the machine learning pipeline.
    
    All transformers in the mod package should inherit from this class to ensure
    consistent behavior and proper target variable handling.
    
    Attributes
    ----------
    logger : logging.Logger
        Logger instance for this transformer
    """
    
    def __init__(self):
        """Initialize the transformer with logging support."""
        super().__init__()
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X while preserving target variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features)
            Transformed array.
        y_new : ndarray array of shape (n_samples,) or (n_samples, n_outputs)
            Transformed target values (None for unsupervised transformations).
        """
        self.logger.debug(f"fit_transform called on {self.__class__.__name__} with X shape: {getattr(X, 'shape', 'unknown')}")
        
        try:
            # non-optimized default implementation; override when a better
            # method is possible for a given clustering algorithm
            if y is None:
                # fit method of arity 1 (unsupervised transformation)
                result = self.fit(X, **fit_params).transform(X, y)
                self.logger.debug(f"fit_transform completed (unsupervised) - X shape: {getattr(result[0], 'shape', 'unknown')}")
                return result
            else:
                # fit method of arity 2 (supervised transformation)
                result = self.fit(X, y, **fit_params).transform(X, y)
                self.logger.debug(f"fit_transform completed (supervised) - X shape: {getattr(result[0], 'shape', 'unknown')}, y shape: {getattr(result[1], 'shape', 'unknown')}")
                return result
        except Exception as e:
            self.logger.error(f"Error in fit_transform for {self.__class__.__name__}: {str(e)}")
            raise
