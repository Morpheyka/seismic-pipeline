"""
Target-aware cross-validation utilities.

This module provides target-aware versions of sklearn's cross-validation utilities
that properly handle (X, y) pairs throughout the cross-validation process.
"""

import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut, 
    ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit,
    cross_val_score, cross_validate, cross_val_predict
)
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils import check_X_y, indexable
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import check_scoring
from .sklearnbaseyt import TransformerMixinYt

# =============================================================================
# Target-Aware Cross-Validation Splitters
# =============================================================================

class KFoldYt(KFold):
    """Target-aware K-Fold cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

class StratifiedKFoldYt(StratifiedKFold):
    """Target-aware Stratified K-Fold cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

class LeaveOneOutYt(LeaveOneOut):
    """Target-aware Leave-One-Out cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

class LeavePOutYt(LeavePOut):
    """Target-aware Leave-P-Out cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

class ShuffleSplitYt(ShuffleSplit):
    """Target-aware Shuffle Split cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

class StratifiedShuffleSplitYt(StratifiedShuffleSplit):
    """Target-aware Stratified Shuffle Split cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

class TimeSeriesSplitYt(TimeSeriesSplit):
    """Target-aware Time Series Split cross-validator that preserves target variables."""
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Call parent split method
        for train_idx, test_idx in super().split(X, y, groups):
            yield train_idx, test_idx

# =============================================================================
# Target-Aware Cross-Validation Functions
# =============================================================================

def cross_val_score_yt(estimator, X, y=None, groups=None, scoring=None, cv=None, 
                      n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      error_score=np.nan, return_train_score=False):
    """
    Target-aware cross-validation score evaluation.
    
    This function evaluates a target-aware estimator by cross-validation and
    returns the scores for each fold.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        Whether to return train scores.
    
    Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
    # Use default CV if none provided
    if cv is None:
        if is_classifier(estimator):
            cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
    
    # Call sklearn's cross_val_score with target-aware CV
    return cross_val_score(
        estimator, X, y, groups=groups, scoring=scoring, cv=cv,
        n_jobs=n_jobs, verbose=verbose, fit_params=fit_params,
        pre_dispatch=pre_dispatch, error_score=error_score
    )

def cross_validate_yt(estimator, X, y=None, groups=None, scoring=None, cv=None,
                     n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                     return_train_score=False, return_estimator=False, error_score=np.nan):
    """
    Target-aware cross-validation evaluation.
    
    This function evaluates a target-aware estimator by cross-validation and
    returns a dictionary with detailed results.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution.
    return_train_score : bool, default=False
        Whether to return train scores.
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    
    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Dictionary with keys as column headers and values as columns, that can be
        imported into a pandas DataFrame.
    """
    # Use default CV if none provided
    if cv is None:
        if is_classifier(estimator):
            cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
    
    # Call sklearn's cross_validate with target-aware CV
    return cross_validate(
        estimator, X, y, groups=groups, scoring=scoring, cv=cv,
        n_jobs=n_jobs, verbose=verbose, fit_params=fit_params,
        pre_dispatch=pre_dispatch, return_train_score=return_train_score,
        return_estimator=return_estimator, error_score=error_score
    )

def cross_val_predict_yt(estimator, X, y=None, groups=None, cv=None, n_jobs=None,
                        verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                        method='predict', error_score=np.nan):
    """
    Target-aware cross-validation prediction.
    
    This function generates cross-validated predictions for a target-aware estimator.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution.
    method : str, default='predict'
        Invokes the passed method name of the passed estimator.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    
    Returns
    -------
    predictions : ndarray
        The predictions.
    """
    # Use default CV if none provided
    if cv is None:
        if is_classifier(estimator):
            cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
    
    # Call sklearn's cross_val_predict with target-aware CV
    return cross_val_predict(
        estimator, X, y, groups=groups, cv=cv, n_jobs=n_jobs,
        verbose=verbose, fit_params=fit_params, pre_dispatch=pre_dispatch,
        method=method, error_score=error_score
    )

# =============================================================================
# Target-Aware Cross-Validation with Target Preservation
# =============================================================================

def cross_val_score_with_targets(estimator, X, y=None, groups=None, scoring=None, cv=None,
                                n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                                error_score=np.nan, return_train_score=False, return_targets=False):
    """
    Target-aware cross-validation that returns both scores and target information.
    
    This function evaluates a target-aware estimator by cross-validation and
    returns the scores along with target information for each fold.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset.
    scoring : str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        Whether to return train scores.
    return_targets : bool, default=False
        Whether to return target information for each fold.
    
    Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    target_info : dict, optional
        Dictionary containing target information for each fold if return_targets=True.
    """
    from sklearn.model_selection import _validation
    
    # Use default CV if none provided
    if cv is None:
        if is_classifier(estimator):
            cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
    
    # Get the cross-validation scores
    scores = cross_val_score_yt(
        estimator, X, y, groups=groups, scoring=scoring, cv=cv,
        n_jobs=n_jobs, verbose=verbose, fit_params=fit_params,
        pre_dispatch=pre_dispatch, error_score=error_score,
        return_train_score=return_train_score
    )
    
    if not return_targets:
        return scores
    
    # Collect target information for each fold
    target_info = {
        'train_targets': [],
        'test_targets': [],
        'train_indices': [],
        'test_indices': []
    }
    
    # Get indices for each fold
    for train_idx, test_idx in cv.split(X, y, groups):
        target_info['train_indices'].append(train_idx)
        target_info['test_indices'].append(test_idx)
        
        if y is not None:
            target_info['train_targets'].append(y[train_idx])
            target_info['test_targets'].append(y[test_idx])
    
    return scores, target_info

# =============================================================================
# Target-Aware Learning Curves
# =============================================================================

def learning_curve_yt(estimator, X, y=None, groups=None, train_sizes=np.linspace(0.1, 1.0, 5),
                     cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=None,
                     pre_dispatch='all', verbose=0, shuffle=False, random_state=None,
                     error_score=np.nan, return_times=False):
    """
    Target-aware learning curve generation.
    
    This function generates learning curves for a target-aware estimator,
    showing how the validation and training scores vary with the number of samples.
    
    Parameters
    ----------
    estimator : estimator object
        An object of that type which implements the methods `fit` and `predict`.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target relative to X for classification or regression.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
    train_sizes : array-like of shape (n_ticks,), default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or a scorer callable object.
    exploit_incremental_learning : bool, default=False
        If the estimator supports incremental learning, this will be used to speed
        up fitting for different training set sizes.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    pre_dispatch : int or str, default='all'
        Number of jobs dispatched in parallel.
    verbose : int, default=0
        Controls the verbosity.
    shuffle : bool, default=False
        Whether to shuffle training data before taking prefixes of it
        based on ``train_sizes``.
    random_state : int, RandomState instance or None, default=None
        Used to shuffle the training data.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_times : bool, default=False
        Whether to return the fit and score times.
    
    Returns
    -------
    train_sizes_abs : array of shape (n_ticks,)
        Numbers of training examples that have been used to generate the
        learning curve.
    train_scores : array of shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores : array of shape (n_ticks, n_cv_folds)
        Scores on test set.
    fit_times : array of shape (n_ticks, n_cv_folds), optional
        Times spent for fitting in seconds.
    score_times : array of shape (n_ticks, n_cv_folds), optional
        Times spent for scoring in seconds.
    """
    from sklearn.model_selection import learning_curve
    
    # Use default CV if none provided
    if cv is None:
        if is_classifier(estimator):
            cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
    
    # Call sklearn's learning_curve with target-aware CV
    return learning_curve(
        estimator, X, y, groups=groups, train_sizes=train_sizes, cv=cv,
        scoring=scoring, exploit_incremental_learning=exploit_incremental_learning,
        n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose,
        shuffle=shuffle, random_state=random_state, error_score=error_score,
        return_times=return_times
    )

def validation_curve_yt(estimator, X, y=None, groups=None, param_name=None, param_range=None,
                       cv=None, scoring=None, n_jobs=None, pre_dispatch='all', verbose=0,
                       error_score=np.nan):
    """
    Target-aware validation curve generation.
    
    This function generates validation curves for a target-aware estimator,
    showing how the validation and training scores vary with a parameter.
    
    Parameters
    ----------
    estimator : estimator object
        An object of that type which implements the methods `fit` and `predict`.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target relative to X for classification or regression.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
    param_name : str
        Name of the parameter that will be varied.
    param_range : array-like of shape (n_values,)
        The values of the parameter that will be evaluated.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or a scorer callable object.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    pre_dispatch : int or str, default='all'
        Number of jobs dispatched in parallel.
    verbose : int, default=0
        Controls the verbosity.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    
    Returns
    -------
    train_scores : array of shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores : array of shape (n_ticks, n_cv_folds)
        Scores on test set.
    """
    from sklearn.model_selection import validation_curve
    
    # Use default CV if none provided
    if cv is None:
        if is_classifier(estimator):
            cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
    
    # Call sklearn's validation_curve with target-aware CV
    return validation_curve(
        estimator, X, y, groups=groups, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
        verbose=verbose, error_score=error_score
    )
