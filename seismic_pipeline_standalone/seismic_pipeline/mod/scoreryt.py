# coding=utf-8
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from .regressionyt import mean_absolute_percentage_error, bias_error, nees_error

def make_scorer_yt(score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs):
    """
    Make a target-aware scorer from a performance metric or loss function.
    
    This is a wrapper around sklearn's make_scorer that works with target-aware estimators.
    """
    return make_scorer(score_func, greater_is_better=greater_is_better, 
                      needs_proba=needs_proba, needs_threshold=needs_threshold, **kwargs)

def yt_accuracy_scorer(estimator, X, y=None):
    """
    Custom scorer for target-aware pipelines where y is generated.
    The pipeline's `predict` method returns (y_pred, y_true_transformed).
    
    Handles edge cases:
    - Empty arrays: returns 0.0
    - Mismatched sizes: returns 0.0
    - Exceptions: returns 0.0 (to match error_score behavior)
    """
    try:
        # When scoring is used, the pipeline's `predict` method is called.
        # In our case, `PipelineYt.predict` will return (predictions, transformed_y)
        result = estimator.predict(X)
        
        # Handle tuple return (y_pred, y_true)
        if isinstance(result, tuple) and len(result) == 2:
            y_pred, y_true = result
        else:
            # If not tuple, assume result is y_pred and we need y_true from somewhere else
            y_pred = result
            y_true = y if y is not None else None
        
        # Validate inputs
        if y_pred is None or y_true is None:
            return 0.0
        
        # Convert to arrays if needed
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        # Check for empty arrays
        if len(y_pred) == 0 or len(y_true) == 0:
            return 0.0
        
        # Check for size mismatch
        if len(y_pred) != len(y_true):
            return 0.0
        
        # Calculate accuracy score
        score = accuracy_score(y_true, y_pred)
        
        # Check for NaN or invalid scores
        if np.isnan(score) or not np.isfinite(score):
            return 0.0
        
        return score
        
    except Exception as e:
        # Return 0.0 for any exception to match error_score=0.0 behavior
        # This prevents NaN scores from propagating
        return 0.0

# Pre-defined scorers
mape_score = make_scorer_yt(mean_absolute_percentage_error, greater_is_better=False)
mse_score = make_scorer_yt(mean_squared_error, greater_is_better=False)
r2_score_yt = make_scorer_yt(r2_score, greater_is_better=True)
bias_score = make_scorer_yt(bias_error, greater_is_better=False)
nees_score = make_scorer_yt(nees_error, greater_is_better=False)
