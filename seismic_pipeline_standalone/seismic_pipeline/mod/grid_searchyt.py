"""
Target-aware hyperparameter tuning utilities.

This module provides target-aware versions of sklearn's hyperparameter tuning utilities
that properly handle (X, y) pairs throughout the search process.
"""

import numpy as np
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler
)
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils import check_X_y, indexable
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import check_scoring
from sklearn.model_selection._search import BaseSearchCV
from .sklearnbaseyt import TransformerMixinYt
from .cross_validationyt import KFoldYt, StratifiedKFoldYt
from .pipelineyt import PipelineYt

# =============================================================================
# Helpers for calculate_best_params_metrics
# =============================================================================


def _get_transformed_data(estimator, X, y):
    """Extract preprocessing steps, run transform, return (X_transformed, y_transformed)."""
    preprocessing_steps = estimator.steps[:-1] if hasattr(estimator, 'steps') else []
    if preprocessing_steps:
        preprocessing_pipeline = PipelineYt(preprocessing_steps)
        X_transformed, y_transformed = preprocessing_pipeline.transform(X, y)
        if y_transformed is None:
            raise ValueError("Could not extract y from pipeline")
    else:
        X_transformed = X
        y_transformed = y if y is not None else None
        if y_transformed is None:
            raise ValueError("y is None and cannot be extracted")
    return X_transformed, y_transformed


def _get_cv_splitter_for_metrics(cv, n_splits, is_clf):
    """Resolve int cv to StratifiedKFoldYt/KFoldYt."""
    if isinstance(cv, int):
        n = cv
        if is_clf:
            return StratifiedKFoldYt(n_splits=n, shuffle=True, random_state=42)
        return KFoldYt(n_splits=n, shuffle=True, random_state=42)
    return cv


def _compute_single_fold_metrics(classifier, X_train, X_test, y_train, y_test):
    """Compute accuracy, precision per class, recall, roc_auc for one fold."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, roc_auc_score
    )
    y_pred = classifier.predict(X_test)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_test = np.asarray(y_test).flatten()
    y_pred = np.asarray(y_pred).flatten()

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)

    try:
        precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        if len(precision) >= 2:
            metrics['precision_class_0'] = float(precision[0])
            metrics['precision_class_1'] = float(precision[1])
        elif len(precision) == 1:
            if 0 in y_test:
                metrics['precision_class_0'] = precision[0] if (y_pred[y_test == 0] == 0).sum() > 0 else 0.0
                metrics['precision_class_1'] = 0.0
            else:
                metrics['precision_class_0'] = 0.0
                metrics['precision_class_1'] = float(precision[0])
        else:
            metrics['precision_class_0'] = 0.0
            metrics['precision_class_1'] = 0.0
    except Exception:
        metrics['precision_class_0'] = 0.0
        metrics['precision_class_1'] = 0.0

    try:
        metrics['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    except Exception:
        metrics['recall'] = 0.0

    try:
        y_proba = classifier.predict_proba(X_test)
        if isinstance(y_proba, tuple):
            y_proba = y_proba[0]
        if y_proba.ndim > 1 and y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        if len(np.unique(y_test)) >= 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba_pos)
        else:
            metrics['roc_auc'] = 0.0
    except Exception:
        metrics['roc_auc'] = 0.0

    return metrics


def _aggregate_fold_metrics(fold_metrics):
    """Compute mean/std per metric from list of fold metric dicts."""
    results = {}
    metric_names = ['accuracy', 'precision_class_0', 'precision_class_1', 'recall', 'roc_auc']
    for name in metric_names:
        values = [m[name] for m in fold_metrics if name in m]
        if values:
            results[f'{name}_mean'] = np.mean(values)
            results[f'{name}_std'] = np.std(values)
        else:
            results[f'{name}_mean'] = 0.0
            results[f'{name}_std'] = 0.0
    return results


# =============================================================================
# Target-Aware Grid Search
# =============================================================================

class GridSearchCVYt(GridSearchCV):
    """Target-aware GridSearchCV that preserves target variables."""
    
    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        """
        Target-aware GridSearchCV constructor.
        
        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (str) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored.
        scoring : str, callable, list/tuple or dict, default=None
            A single str (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.
        n_jobs : int, default=None
            Number of jobs to run in parallel.
        refit : bool, default=True
            Refit an estimator using the best found parameters on the whole
            dataset.
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
        verbose : int, default=0
            Controls the verbosity.
        pre_dispatch : int or str, default='2*n_jobs'
            Controls the number of jobs that get dispatched during parallel
            execution.
        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
        return_train_score : bool, default=False
            If False, the cv_results_ attribute will not include training scores.
        """
        # Use target-aware CV if none provided
        if cv is None:
            if is_classifier(estimator):
                cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
        
        super(GridSearchCVYt, self).__init__(
            estimator=estimator, param_grid=param_grid, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score
        )
    
    def calculate_best_params_metrics(self, X, y=None):
        """
        Calculate comprehensive metrics for the best parameters across all CV folds.

        This method re-runs cross-validation with the best found parameters and
        calculates accuracy, precision (per class), recall (macro-averaged), and ROC-AUC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,), default=None
            Target vector.

        Returns
        -------
        metrics : dict
            Dictionary with metric names as keys and (mean, std) tuples as values:
            - 'accuracy_mean', 'accuracy_std'
            - 'precision_class_0_mean', 'precision_class_0_std'
            - 'precision_class_1_mean', 'precision_class_1_std'
            - 'recall_mean', 'recall_std'
            - 'roc_auc_mean', 'roc_auc_std'
        """
        check_is_fitted(self)

        best_estimator = self.best_estimator_
        X_transformed, y_transformed = _get_transformed_data(best_estimator, X, y)

        n_splits = self.cv if isinstance(self.cv, int) else getattr(self.cv, 'n_splits', None)
        cv = _get_cv_splitter_for_metrics(
            self.cv, n_splits or 5, is_clf=is_classifier(best_estimator)
        )

        # Extract accuracy from grid search cv_results_ when available (authoritative)
        accuracy_scores_from_grid = None
        accuracy_std_from_grid = None
        if (hasattr(self, 'cv_results_') and hasattr(self, 'best_index_')
                and self.best_index_ is not None):
            best_idx = self.best_index_
            if 'std_test_score' in self.cv_results_ and best_idx < len(self.cv_results_['std_test_score']):
                accuracy_std_from_grid = self.cv_results_['std_test_score'][best_idx]
            if n_splits:
                fold_accuracy_scores = []
                for i in range(n_splits):
                    fold_key = f'split{i}_test_score'
                    if fold_key in self.cv_results_ and best_idx < len(self.cv_results_[fold_key]):
                        fold_accuracy_scores.append(self.cv_results_[fold_key][best_idx])
                if len(fold_accuracy_scores) == n_splits:
                    accuracy_scores_from_grid = fold_accuracy_scores
                    if accuracy_std_from_grid is None and len(fold_accuracy_scores) > 1:
                        accuracy_std_from_grid = np.std(fold_accuracy_scores)

        classifier = (best_estimator.named_steps['classifier']
                      if hasattr(best_estimator, 'steps') else best_estimator)

        fold_metrics_list = []
        for fold_num, (train_idx_trans, test_idx_trans) in enumerate(cv.split(X_transformed, y_transformed)):
            X_train_trans = X_transformed[train_idx_trans]
            X_test_trans = X_transformed[test_idx_trans]
            y_train_trans = y_transformed[train_idx_trans]
            y_test_trans = y_transformed[test_idx_trans]

            classifier.fit(X_train_trans, y_train_trans)
            metrics = _compute_single_fold_metrics(
                classifier, X_train_trans, X_test_trans, y_train_trans, y_test_trans
            )
            if accuracy_scores_from_grid is not None and fold_num < len(accuracy_scores_from_grid):
                metrics['accuracy'] = accuracy_scores_from_grid[fold_num]
            fold_metrics_list.append(metrics)

        results = _aggregate_fold_metrics(fold_metrics_list)

        if hasattr(self, 'best_score_'):
            results['accuracy_mean'] = self.best_score_
        if accuracy_std_from_grid is not None:
            results['accuracy_std'] = accuracy_std_from_grid
        elif accuracy_scores_from_grid is not None and len(accuracy_scores_from_grid) > 1:
            results['accuracy_std'] = np.std(accuracy_scores_from_grid)

        return results

class RandomizedSearchCVYt(RandomizedSearchCV):
    """Target-aware RandomizedSearchCV that preserves target variables."""
    
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None, error_score=np.nan,
                 return_train_score=False):
        """
        Target-aware RandomizedSearchCV constructor.
        
        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.
        param_distributions : dict
            Dictionary with parameters names (str) as keys and distributions
            or lists of parameters to try.
        n_iter : int, default=10
            Number of parameter settings that are sampled.
        scoring : str, callable, list/tuple or dict, default=None
            A single str (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.
        n_jobs : int, default=None
            Number of jobs to run in parallel.
        refit : bool, default=True
            Refit an estimator using the best found parameters on the whole
            dataset.
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
        verbose : int, default=0
            Controls the verbosity.
        pre_dispatch : int or str, default='2*n_jobs'
            Controls the number of jobs that get dispatched during parallel
            execution.
        random_state : int, RandomState instance or None, default=None
            Pseudo random number generator state used for random uniform sampling.
        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
        return_train_score : bool, default=False
            If False, the cv_results_ attribute will not include training scores.
        """
        # Use target-aware CV if none provided
        if cv is None:
            if is_classifier(estimator):
                cv = StratifiedKFoldYt(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFoldYt(n_splits=5, shuffle=True, random_state=42)
        
        super(RandomizedSearchCVYt, self).__init__(
            estimator=estimator, param_distributions=param_distributions,
            n_iter=n_iter, scoring=scoring, n_jobs=n_jobs,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            random_state=random_state, error_score=error_score,
            return_train_score=return_train_score
        )

# =============================================================================
# Target-Aware Parameter Search Utilities
# =============================================================================

def grid_search_yt(estimator, param_grid, X, y=None, groups=None, scoring=None, cv=None,
                  n_jobs=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                  return_train_score=False, refit=True):
    """
    Target-aware grid search function.
    
    This function performs a grid search over specified parameter values for a
    target-aware estimator.
    
    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of
        parameter settings to try as values.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target relative to X for classification or regression.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset.
    scoring : str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    
    Returns
    -------
    grid_search : GridSearchCVYt
        Fitted GridSearchCVYt object.
    """
    # Create target-aware grid search
    grid_search = GridSearchCVYt(
        estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv,
        n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch,
        error_score=error_score, return_train_score=return_train_score, refit=refit
    )
    
    # Fit the grid search
    grid_search.fit(X, y, groups=groups)
    
    return grid_search

def random_search_yt(estimator, param_distributions, X, y=None, groups=None, n_iter=10,
                    scoring=None, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                    random_state=None, error_score=np.nan, return_train_score=False, refit=True):
    """
    Target-aware random search function.
    
    This function performs a random search over specified parameter distributions
    for a target-aware estimator.
    
    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    param_distributions : dict
        Dictionary with parameters names (str) as keys and distributions
        or lists of parameters to try.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target relative to X for classification or regression.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset.
    n_iter : int, default=10
        Number of parameter settings that are sampled.
    scoring : str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    
    Returns
    -------
    random_search : RandomizedSearchCVYt
        Fitted RandomizedSearchCVYt object.
    """
    # Create target-aware random search
    random_search = RandomizedSearchCVYt(
        estimator=estimator, param_distributions=param_distributions, n_iter=n_iter,
        scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose,
        pre_dispatch=pre_dispatch, random_state=random_state,
        error_score=error_score, return_train_score=return_train_score, refit=refit
    )
    
    # Fit the random search
    random_search.fit(X, y, groups=groups)
    
    return random_search

# =============================================================================
# Target-Aware Parameter Grid Utilities
# =============================================================================

def create_param_grid_yt(estimator_type='pipeline', task_type='regression'):
    """
    Create parameter grids for target-aware estimators.
    
    This function creates common parameter grids for different types of
    target-aware estimators and tasks.
    
    Parameters
    ----------
    estimator_type : str, default='pipeline'
        Type of estimator ('pipeline', 'transformer', 'classifier', 'regressor').
    task_type : str, default='regression'
        Type of task ('regression', 'classification').
    
    Returns
    -------
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values.
    """
    param_grid = {}
    
    if estimator_type == 'pipeline':
        # Common pipeline parameters
        param_grid.update({
            'scaler__regression': [True, False],
            'scaler__copy': [True, False],
        })
        
        # Add transformer-specific parameters
        param_grid.update({
            'poly__degree': [1, 2, 3],
            'poly__interaction_only': [True, False],
            'poly__include_bias': [True, False],
        })
        
        # Add feature selection parameters
        param_grid.update({
            'selector__n_features_to_select': [5, 10, 15, 20],
            'selector__threshold': [0.01, 0.1, 0.2, 0.5],
        })
        
        # Add decomposition parameters
        param_grid.update({
            'pca__n_components': [0.8, 0.9, 0.95, 0.99],
            'pca__whiten': [True, False],
        })
        
    elif estimator_type == 'transformer':
        # Transformer-specific parameters
        param_grid.update({
            'regression': [True, False],
            'copy': [True, False],
        })
        
    elif estimator_type == 'classifier':
        # Classification-specific parameters
        if task_type == 'classification':
            param_grid.update({
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
            })
        else:
            param_grid.update({
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
            })
    
    elif estimator_type == 'regressor':
        # Regression-specific parameters
        param_grid.update({
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
        })
    
    return param_grid

def create_param_distributions_yt(estimator_type='pipeline', task_type='regression'):
    """
    Create parameter distributions for target-aware estimators.
    
    This function creates common parameter distributions for different types of
    target-aware estimators and tasks.
    
    Parameters
    ----------
    estimator_type : str, default='pipeline'
        Type of estimator ('pipeline', 'transformer', 'classifier', 'regressor').
    task_type : str, default='regression'
        Type of task ('regression', 'classification').
    
    Returns
    -------
    param_distributions : dict
        Dictionary with parameter names as keys and distributions or lists.
    """
    from scipy.stats import uniform, randint, loguniform
    
    param_distributions = {}
    
    if estimator_type == 'pipeline':
        # Common pipeline parameters
        param_distributions.update({
            'scaler__regression': [True, False],
            'scaler__copy': [True, False],
        })
        
        # Add transformer-specific parameters
        param_distributions.update({
            'poly__degree': randint(1, 4),
            'poly__interaction_only': [True, False],
            'poly__include_bias': [True, False],
        })
        
        # Add feature selection parameters
        param_distributions.update({
            'selector__n_features_to_select': randint(5, 25),
            'selector__threshold': uniform(0.01, 0.5),
        })
        
        # Add decomposition parameters
        param_distributions.update({
            'pca__n_components': uniform(0.8, 0.2),
            'pca__whiten': [True, False],
        })
        
    elif estimator_type == 'transformer':
        # Transformer-specific parameters
        param_distributions.update({
            'regression': [True, False],
            'copy': [True, False],
        })
        
    elif estimator_type == 'classifier':
        # Classification-specific parameters
        if task_type == 'classification':
            param_distributions.update({
                'n_estimators': randint(50, 300),
                'max_depth': [None] + list(randint(5, 50).rvs(10)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
            })
        else:
            param_distributions.update({
                'n_estimators': randint(50, 300),
                'max_depth': [None] + list(randint(5, 50).rvs(10)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
            })
    
    elif estimator_type == 'regressor':
        # Regression-specific parameters
        param_distributions.update({
            'n_estimators': randint(50, 300),
            'max_depth': [None] + list(randint(5, 50).rvs(10)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
        })
    
    return param_distributions

# =============================================================================
# Target-Aware Hyperparameter Tuning with Target Preservation
# =============================================================================

def tune_hyperparameters_yt(estimator, X, y=None, groups=None, param_grid=None,
                           param_distributions=None, search_type='grid', n_iter=10,
                           scoring=None, cv=None, n_jobs=None, verbose=0,
                           pre_dispatch='2*n_jobs', random_state=None,
                           error_score=np.nan, return_train_score=False, refit=True):
    """
    Target-aware hyperparameter tuning function.
    
    This function performs hyperparameter tuning for a target-aware estimator
    using either grid search or random search.
    
    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target relative to X for classification or regression.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset.
    param_grid : dict or list of dictionaries, default=None
        Dictionary with parameters names (str) as keys and lists of
        parameter settings to try as values.
    param_distributions : dict, default=None
        Dictionary with parameters names (str) as keys and distributions
        or lists of parameters to try.
    search_type : str, default='grid'
        Type of search ('grid' or 'random').
    n_iter : int, default=10
        Number of parameter settings that are sampled (for random search).
    scoring : str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    
    Returns
    -------
    search_results : GridSearchCVYt or RandomizedSearchCVYt
        Fitted search object.
    """
    if search_type == 'grid':
        if param_grid is None:
            # Create default parameter grid
            param_grid = create_param_grid_yt()
        
        # Perform grid search
        search_results = grid_search_yt(
            estimator=estimator, param_grid=param_grid, X=X, y=y, groups=groups,
            scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score, refit=refit
        )
    
    elif search_type == 'random':
        if param_distributions is None:
            # Create default parameter distributions
            param_distributions = create_param_distributions_yt()
        
        # Perform random search
        search_results = random_search_yt(
            estimator=estimator, param_distributions=param_distributions,
            X=X, y=y, groups=groups, n_iter=n_iter, scoring=scoring, cv=cv,
            n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch,
            random_state=random_state, error_score=error_score,
            return_train_score=return_train_score, refit=refit
        )
    
    else:
        raise ValueError("search_type must be 'grid' or 'random'")
    
    return search_results

# =============================================================================
# Target-Aware Hyperparameter Tuning with Multiple Metrics
# =============================================================================

def tune_hyperparameters_multi_metric_yt(estimator, X, y=None, groups=None, param_grid=None,
                                        param_distributions=None, search_type='grid', n_iter=10,
                                        scoring=None, cv=None, n_jobs=None, verbose=0,
                                        pre_dispatch='2*n_jobs', random_state=None,
                                        error_score=np.nan, return_train_score=False, refit=True):
    """
    Target-aware hyperparameter tuning with multiple metrics.
    
    This function performs hyperparameter tuning for a target-aware estimator
    using multiple scoring metrics.
    
    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target relative to X for classification or regression.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset.
    param_grid : dict or list of dictionaries, default=None
        Dictionary with parameters names (str) as keys and lists of
        parameter settings to try as values.
    param_distributions : dict, default=None
        Dictionary with parameters names (str) as keys and distributions
        or lists of parameters to try.
    search_type : str, default='grid'
        Type of search ('grid' or 'random').
    n_iter : int, default=10
        Number of parameter settings that are sampled (for random search).
    scoring : str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    
    Returns
    -------
    search_results : GridSearchCVYt or RandomizedSearchCVYt
        Fitted search object with multiple metrics.
    """
    # Use default scoring if none provided
    if scoring is None:
        if is_classifier(estimator):
            scoring = ['accuracy', 'precision', 'recall', 'f1']
        else:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    # Perform hyperparameter tuning
    search_results = tune_hyperparameters_yt(
        estimator=estimator, X=X, y=y, groups=groups, param_grid=param_grid,
        param_distributions=param_distributions, search_type=search_type,
        n_iter=n_iter, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose,
        pre_dispatch=pre_dispatch, random_state=random_state,
        error_score=error_score, return_train_score=return_train_score, refit=refit
    )
    
    return search_results
