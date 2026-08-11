# coding=utf-8
"""
Utility functions for the seismic event pipeline.

This module contains helper functions that are used across different parts
of the pipeline for data processing and debugging.
"""

import os
import pandas as pd
import numpy as np

# Re-export from threading_config (no numpy) - import from threading_config
# directly if you need to run configure_threading before numpy
from .threading_config import configure_threading


def save_step_data(X, y, step_name, output_dir, output_prefix, step_number):
    """
    Save dataset at each pipeline step for verification.
    
    Args:
        X: Input data (can be list of dicts, numpy array, or other formats)
        y: Target data (numpy array or None)
        step_name: Name of the pipeline step
        output_dir: Directory to save files
        output_prefix: Prefix for output files
        step_number: Step number for ordering
    
    Returns:
        tuple: (X_df, y_df) - DataFrames created from the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert X to DataFrame if it's a list of dicts
    if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
        # Extract relevant information for DataFrame
        X_data = []
        for item in X:
            if 'window_dates' in item:
                # Create a row with rat_id, event_date, and window_dates
                row = {
                    'rat_id': item.get('rat_id', ''),
                    'original_event_date': item.get('original_event_date', ''),
                    'window_type': item.get('window_type', ''),
                    'window_dates': str(item.get('window_dates', [])),
                    'original_rat_id': item.get('original_rat_id', '')
                }
                X_data.append(row)
        X_df = pd.DataFrame(X_data)
    elif isinstance(X, np.ndarray):
        # Handle numpy arrays
        if X.ndim == 1:
            # 1D array - convert to 2D
            X_df = pd.DataFrame(X.reshape(-1, 1), columns=['value'])
        else:
            # 2D array
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    else:
        # Handle other formats
        try:
            if hasattr(X, 'shape') and len(X.shape) > 0:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            else:
                X_df = pd.DataFrame(X)
        except (TypeError, ValueError):
            # Fallback - convert to string representation
            X_df = pd.DataFrame({'data': [str(X)]})
    
    # Convert y to DataFrame
    if y is not None:
        if isinstance(y, np.ndarray):
            y_df = pd.DataFrame(y, columns=['target'])
        else:
            y_df = pd.DataFrame(y, columns=['target'])
    else:
        y_df = pd.DataFrame()
    
    # Save X and y separately
    X_filename = os.path.join(output_dir, f'{output_prefix}_step_{step_number:02d}_{step_name}_X.csv')
    y_filename = os.path.join(output_dir, f'{output_prefix}_step_{step_number:02d}_{step_name}_y.csv')
    
    X_df.to_csv(X_filename, index=False)
    if not y_df.empty:
        y_df.to_csv(y_filename, index=False)
    
    print(f"  Step {step_number} ({step_name}) - X shape: {X_df.shape}, y shape: {y_df.shape}")
    print(f"    X saved to: {X_filename}")
    if not y_df.empty:
        print(f"    y saved to: {y_filename}")
    
    return X_df, y_df
