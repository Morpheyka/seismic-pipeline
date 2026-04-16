#!/usr/bin/env python3
"""
Hyperparameter Grid Search Visualization Module

This module provides visualization functions for scikit-learn GridSearchCV results.
It creates comprehensive multi-dimensional slices showing how different hyperparameter
combinations affect model performance.

Usage:
    from seismic_pipeline.visualization import visualize_hyperparameter_grid_slices
    
    # After running GridSearchCV
    grid_search = GridSearchCV(estimator, param_grid, ...)
    grid_search.fit(X, y)
    
    # Generate visualizations for all parameter pairs
    visualize_hyperparameter_grid_slices(grid_search, output_dir='./results')
    
    # Or visualize only specific parameter pairs
    visualize_hyperparameter_grid_slices(
        grid_search, 
        output_dir='./results',
        param_pairs=[('param_C', 'param_gamma'), ('param_kernel', 'param_C')]
    )

Functions:
    - visualize_hyperparameter_grid_slices: Main function to create visualizations
    - build_slices: Build visualization slices for hyperparameter grid search results
    - autoslice: Auto-generate slices for multi-dimensional parameter visualization
    - build_imshow: Build imshow visualization for parameter grid
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
import re
import hashlib
import gc
import os


def _save_figure(fig, feature, result_path):
    """
    Save a matplotlib figure ensuring a filesystem-safe filename and hashed fallback.
    
    Args:
        fig: Matplotlib figure object
        feature: Feature identifier used for naming the file
        result_path: Directory path where the figure should be saved
    
    Returns:
        The path to the saved figure (with .png extension).
    """
    feature = feature if feature else 'Title'
    if isinstance(feature, tuple):
        feature_str = "_".join(map(str, feature))
    else:
        feature_str = str(feature)
    safe_feature = re.sub(r'[\/\\\.\(\)\s]', '_', feature_str)
    file_base = os.path.join(result_path, safe_feature)
    
    try:
        fig.savefig(fname=file_base, dpi=300, bbox_inches='tight')
        return f"{file_base}.png"
    except OSError:
        print('Filename too long. Using abbreviated version with hash.')
        if isinstance(feature, tuple):
            feature_str = "_".join([str(item)[0] for item in feature])
        safe_feature_clean = re.sub(r'[\/\\\.\(\)\s]', '_', feature_str)
        feature_hash = hashlib.md5(safe_feature_clean.encode()).hexdigest()
        hashed_feature = f"{safe_feature_clean[:30]}_{feature_hash[:8]}"
        file_base = os.path.join(result_path, hashed_feature)
        fig.savefig(fname=file_base, dpi=300, bbox_inches='tight')
        return f"{file_base}.png"


def build_imshow(ax, data, ticks, labels, feature, longest_dim):
    """
    Build imshow visualization for parameter grid.
    
    Args:
        ax: Matplotlib axis object
        data: 2D numpy array of scores
        ticks: List of [xticks, yticks] values
        labels: List of [xlabel, ylabel] strings
        feature: Feature name for title
        longest_dim: Longest dimension for font size calculation
    """
    xticks, yticks = ticks
    xlabel, ylabel = labels
    
    # Transpose if needed to make visualization clearer
    if data.shape[1] > data.shape[0]:
        data = np.transpose(data)
    if len(xticks) > len(yticks):
        xticks, yticks = yticks, xticks
        xlabel, ylabel = ylabel, xlabel
    
    # Create heatmap
    im = ax.imshow(data, cmap='Purples', vmin=0)
    
    xticklabels = xticks
    yticklabels = yticks

    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))

    # Calculate font size based on dimensions (keep labels compact)
    base_size = 6.0
    dynamic_size = 7.5 * longest_dim / max(len(yticks), 1)
    fsize = max(5.5, min(9.0, dynamic_size))
    ax.set_xticklabels(xticklabels, fontsize=fsize, rotation=30)
    ax.set_yticklabels(yticklabels, fontsize=fsize)

    ax.grid(False)
    
    # Add score values as text
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            ax.text(j, i, round(data[i][j], 2), ha='center', va='center', color='r', fontsize=fsize)
    
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    return None


def autoslice(combinations, max_ind, res_list, rdim, params, ax, feature, longest_dim):
    """
    Auto-generate slices for multi-dimensional parameter visualization.
    
    This function creates 2D slices through a multi-dimensional hyperparameter space,
    fixing certain parameters at their optimal values while varying others.
    
    Args:
        combinations: Array of parameter dimension combinations to visualize
        max_ind: Indices of the maximum score in the parameter grid
        res_list: Multi-dimensional array of scores
        rdim: Number of dimensions in the parameter space
        params: Dictionary of parameter names and values
        ax: 2D array of matplotlib axis objects
        feature: Feature name for titles
        longest_dim: Longest dimension for font size calculation
    """
    for i in range(len(combinations)):
        for j in range(len(combinations[i])):
            current = combinations[i][j]
            
            # Skip empty subplots (marked with -1 or None)
            if current is None or (isinstance(current, np.ndarray) and len(current) > 0 and current[0] == -1):
                ax[i][j].axis('off')
                continue
            
            data = res_list
            axis = 0
            try:
                # Extract 2D slice by fixing parameters at optimal values
                for ind in current:
                    axis += 0 if np.where(current == ind)[0] == 0 else 1
                    # Check if max_ind[ind] has elements
                    if len(max_ind[ind]) > 0:
                        # Check if the axis exists and has the required size
                        if ind - axis < data.ndim and data.shape[ind - axis] > max_ind[ind][0]:
                            data = np.take(data, max_ind[ind][0], ind - axis)
                        else:
                            # Use index 0 if max_ind is out of bounds
                            data = np.take(data, 0, ind - axis)
                    else:
                        # Use index 0 if max_ind is empty
                        data = np.take(data, 0, ind - axis)
                        
                clear_ind = np.delete(np.arange(rdim), current)
                ticks = [list(params.values())[clear_ind[k]] for k in range(-1, -3, -1)]
                labels = [list(params.keys())[clear_ind[k]] for k in range(-1, -3, -1)]
                
                # Handle case where data might be 0-dimensional
                if data.ndim == 0:
                    data = np.array([[data]])
                elif data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                build_imshow(ax[i][j], data, ticks, labels, feature, longest_dim)
            except Exception as e:
                print(f"Warning: Could not create slice at position ({i}, {j}): {e}")
                # Create an empty plot with error message
                ax[i][j].text(0.5, 0.5, f'Slice unavailable\n{str(e)[:50]}', 
                            ha='center', va='center', transform=ax[i][j].transAxes)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                
    del combinations
    return None


def build_slices(res_list, params, score_max, feature, result_path, param_pairs=None,
                verbose: bool = False):
    """
    Build visualization slices for hyperparameter grid search results.
    
    Creates different types of visualizations depending on the dimensionality:
    - 1D: Single parameter bar plot
    - 2D: Single heatmap
    - 3D+: Multiple 2D slices through the parameter space
    
    Args:
        res_list: Multi-dimensional array of scores
        params: Dictionary mapping parameter names to their tested values
        score_max: Maximum score achieved in grid search
        feature: Name for the output file
        result_path: Directory path to save visualizations
        param_pairs: Optional list of tuples specifying which parameter pairs to visualize.
                     Each tuple should contain two parameter names (with 'param_' prefix).
                     If None, all possible pairs will be visualized.
                     Example: [('param_C', 'param_gamma'), ('param_kernel', 'param_C')]
    """
    feature = feature if feature else 'Title'
    def _log(message):
        if verbose:
            print(message)
    plt.ioff()
    rdim = np.ndim(res_list)
    longest_dim = np.max(list(map(len, list(params.values()))))
    
    # Warn if param_pairs is specified for 1D or 2D cases (where it's not applicable)
    if param_pairs is not None and rdim <= 2:
        _log(f"Note: param_pairs parameter is ignored for {rdim}D parameter space. "
             f"It only applies to 3D+ parameter spaces.")
        param_pairs = None
    
    if rdim == 1:
        # Single parameter case
        res_list = np.reshape(res_list, (len(res_list), 1))
        fig, ax = plt.subplots()
        ticks = [params[next(iter(params))], [1]]
        labels = ["", next(iter(params))]
        build_imshow(ax, res_list, ticks, labels, feature, longest_dim)
    elif rdim == 2:
        # Two parameters case
        fig, ax = plt.subplots()
        ticks = [list(params.values())[-1], list(params.values())[-2]]
        labels = list(params.keys())
        build_imshow(ax, res_list, ticks, labels, feature, longest_dim)
    else:
        # Three or more parameters - create multiple 2D slices
        res_list = np.array(res_list)
        
        # Find the maximum score location using the provided score_max
        # This ensures we find the location of the actual best score from grid search
        max_ind = np.where(res_list == score_max)
        
        # If not found, try with nanmax (in case of floating point precision issues)
        if len(max_ind[0]) == 0 or all(len(mi) == 0 for mi in max_ind):
            _log(f"Best score {score_max:.4f} not found in array, trying nanmax...")
            max_ind = np.where(res_list == np.nanmax(res_list))
        
        # Check if we have valid maximum indices
        if len(max_ind[0]) == 0 or all(len(mi) == 0 for mi in max_ind):
            print("Warning: No valid scores found in the grid. Using default indices.")
            max_ind = tuple([np.array([0]) for _ in range(rdim)])
        else:
            _log(f"Found maximum score at indices: {[mi[0] if len(mi) > 0 else 0 for mi in max_ind]}")
        
        # Generate combinations based on param_pairs filter
        param_names_list = list(params.keys())
        
        if param_pairs is not None:
            # Filter to only show requested parameter pairs
            _log(f"Filtering to show only specified parameter pairs: {param_pairs}")
            
            # Validate that all requested parameters exist
            for pair in param_pairs:
                if len(pair) != 2:
                    raise ValueError(f"Each parameter pair must contain exactly 2 parameters. Got: {pair}")
                for param_name in pair:
                    if param_name not in param_names_list:
                        raise ValueError(f"Parameter '{param_name}' not found in grid search results. "
                                       f"Available parameters: {param_names_list}")
            
            # Create mapping from parameter names to their indices
            param_name_to_idx = {name: idx for idx, name in enumerate(param_names_list)}
            
            # Build combinations: for each requested pair, create a combination that fixes all other parameters
            combs = []
            for param1_name, param2_name in param_pairs:
                param1_idx = param_name_to_idx[param1_name]
                param2_idx = param_name_to_idx[param2_name]
                
                # Create a combination that fixes all parameters except these two
                # This means we need to include all other parameter indices in the "fixed" set
                fixed_indices = [idx for idx in range(rdim) if idx != param1_idx and idx != param2_idx]
                combs.append(tuple(fixed_indices))
            
            # Remove duplicates while preserving order
            seen = set()
            combs = [c for c in combs if not (c in seen or seen.add(c))]
            
            _log(f"Generated {len(combs)} unique slice(s) for requested parameter pairs")
        else:
            # Generate all possible 2D combinations of parameters (original behavior)
            combs = list(itr.combinations(np.arange(rdim), rdim - 2))
            _log(f"Generating all possible parameter pair combinations: {len(combs)} slices")

        # Calculate subplot layout
        img_count = len(combs)
        if img_count == 0:
            print("Warning: No combinations to visualize. Skipping visualization.")
            return None
        
        col_count = 3 if img_count % 3 == 0 else 2 if img_count % 2 == 0 else 5 if img_count % 5 == 0 else 7
        row_count = img_count // col_count if img_count % col_count == 0 else (img_count + col_count - 1) // col_count

        fig, ax = plt.subplots(row_count, col_count, figsize=(6*col_count, longest_dim*row_count))
        fig.subplots_adjust(hspace=0.5, wspace=0.4)
        
        # Handle single subplot case
        if row_count == 1 and col_count == 1:
            ax = np.array([[ax]])
        elif row_count == 1:
            ax = ax.reshape(1, -1)
        elif col_count == 1:
            ax = ax.reshape(-1, 1)
        else:
            ax = np.reshape(ax, [row_count, col_count])
        
        # Reshape combinations to match grid layout
        # Convert list of tuples to 3D numpy array: [row_count, col_count, rdim - 2]
        combs_list = list(combs)
        target_size = row_count * col_count
        
        # Pad if needed (for filtered case)
        while len(combs_list) < target_size:
            # Pad with a sentinel value - use a tuple of -1s
            combs_list.append(tuple([-1] * (rdim - 2)))
        
        # Convert to numpy array and reshape
        combs_array = np.array([np.array(c) for c in combs_list[:target_size]])
        combs_grid = combs_array.reshape(row_count, col_count, rdim - 2)
        
        # Generate all slices
        autoslice(combs_grid, max_ind, res_list, rdim, params, ax, feature, longest_dim)
    
    saved_path = _save_figure(fig, feature, result_path)
    plt.close(fig)
    del res_list, fig, ax, params
    gc.collect()
    return saved_path


def visualize_hyperparameter_grid_slices(grid_search, output_dir='.', param_pairs=None,
                                         focus_param_prefixes=None, verbose: bool = False,
                                         feature_name: str = None):
    """
    Visualize hyperparameter grid search results using multi-dimensional slices.
    
    This function creates comprehensive visualizations of scikit-learn GridSearchCV results,
    showing how different hyperparameter combinations affect model performance. For 3+ 
    dimensional parameter spaces, it generates multiple 2D slices through the space,
    fixing some parameters at optimal values while varying others.
    
    Args:
        grid_search: Fitted sklearn GridSearchCV object
        output_dir: Directory path to save visualization (default: current directory)
        param_pairs: Optional list of tuples specifying which parameter pairs to visualize.
                     Each tuple should contain two parameter names (with 'param_' prefix).
                     If None, all possible pairs (or auto-selected focus pairs) will be visualized.
                     Example: [('param_C', 'param_gamma'), ('param_kernel', 'param_C')]
        focus_param_prefixes: Optional list of parameter name prefixes (with 'param_' prefix) to
                              automatically focus on when param_pairs is not provided. Defaults to
                              label generator, REM calculator, and feature extractor parameters.
        verbose: If True, print detailed progress information.
        feature_name: Optional name for the output file (without extension). If None, defaults to
                      "Hyperparameter_Grid_Search". The output file will be saved as "{feature_name}.png"
                      in the output_dir.
    
    Example:
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.svm import SVC
        >>> 
        >>> param_grid = {
        ...     'C': [0.1, 1, 10],
        ...     'kernel': ['rbf', 'linear'],
        ...     'gamma': [0.001, 0.01, 0.1]
        ... }
        >>> grid_search = GridSearchCV(SVC(), param_grid, cv=5)
        >>> grid_search.fit(X_train, y_train)
        >>> 
        >>> # Generate visualizations for all parameter pairs
        >>> from seismic_pipeline.visualization import visualize_hyperparameter_grid_slices
        >>> visualize_hyperparameter_grid_slices(grid_search, output_dir='./results')
        >>> 
        >>> # Or visualize only specific parameter pairs
        >>> visualize_hyperparameter_grid_slices(
        ...     grid_search, 
        ...     output_dir='./results',
        ...     param_pairs=[('param_C', 'param_gamma'), ('param_kernel', 'param_C')]
        ... )
    
    The function will create a PNG file showing:
    - For 1D parameter space: A single bar chart
    - For 2D parameter space: A single heatmap
    - For 3D+ parameter space: Multiple 2D heatmap slices (filtered by param_pairs if specified)
    
    Each visualization shows the mean cross-validation score for different parameter combinations.
    """
    def _log(message):
        if verbose:
            print(message)
    
    _log("=== Creating Hyperparameter Grid Slices Visualizations ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results from grid search
    cv_results = grid_search.cv_results_
    param_names = list(cv_results.keys())
    
    # Filter parameter names (exclude test scores, fit times, etc.)
    param_names = [name for name in param_names if name.startswith('param_')]
    
    if not param_names:
        print("No hyperparameters found in grid search results.")
        return
    
    _log(f"Found hyperparameters: {param_names}")
    
    def _is_masked_value(val):
        return (
            val is None
            or (hasattr(val, '__class__') and 'MaskedConstant' in str(val.__class__))
        )

    param_validity = {
        name: not any(_is_masked_value(v) for v in cv_results[name])
        for name in param_names
    }
    valid_param_names = [name for name in param_names if param_validity[name]]

    if not valid_param_names:
        print("No universally applicable hyperparameters available for visualization.")
        return

    filtered_out = [name for name in param_names if name not in valid_param_names]
    if filtered_out:
        print(f"Skipping classifier-specific parameters for visualization: {filtered_out}")

    param_names = valid_param_names

    # Filter user-provided param pairs to valid ones
    if param_pairs is not None:
        filtered_pairs = [
            pair for pair in param_pairs
            if len(pair) == 2 and all(p in param_names for p in pair)
        ]
        if not filtered_pairs:
            print("Provided param_pairs do not reference universally applicable parameters. Falling back to auto-selection.")
            param_pairs = None
        else:
            param_pairs = filtered_pairs

    # Auto-generate focus parameter pairs if none provided
    if param_pairs is None:
        if focus_param_prefixes is None:
            focus_param_prefixes = [
                'param_label_generator__',
                'param_rem_calculator__',
                'param_feature_extractor__'
            ]
        if focus_param_prefixes:
            relevant_params = [
                name for name in param_names
                if any(name.startswith(prefix) for prefix in focus_param_prefixes)
            ]
            if len(relevant_params) >= 2:
                param_pairs = list(itr.combinations(relevant_params, 2))
                _log(f"Auto-selected {len(param_pairs)} parameter pair(s) using focus prefixes "
                     f"{focus_param_prefixes}: {param_pairs}")
            else:
                _log(f"Auto-selection skipped (found {len(relevant_params)} relevant parameters). "
                     f"Falling back to all parameter combinations.")
                param_pairs = None
    
    # Get unique parameter values
    param_values = {}
    for param_name in param_names:
        values = cv_results[param_name]
        # Handle classifier objects specially
        if param_name == 'param_classifier':
            unique_values = list(set([type(v).__name__ for v in values]))
        else:
            try:
                # Convert to list and filter out None/masked values
                clean_values = [v for v in values if v is not None and not (hasattr(v, '__class__') and 'MaskedConstant' in str(v.__class__))]
                if clean_values:
                    unique_values = sorted(list(set(clean_values)))
                else:
                    unique_values = []
            except (TypeError, ValueError):
                # If sorting fails, just get unique values without sorting
                try:
                    clean_values = [v for v in values if v is not None and not (hasattr(v, '__class__') and 'MaskedConstant' in str(v.__class__))]
                    unique_values = list(set(clean_values)) if clean_values else []
                except:
                    unique_values = []
        param_values[param_name] = unique_values
    
    _log(f"Parameter value ranges: {param_values}")
    
    # Filter out parameters with no valid values
    param_values = {k: v for k, v in param_values.items() if len(v) > 0}
    param_names = list(param_values.keys())
    
    if not param_names:
        print("No valid hyperparameters found for visualization.")
        return
    
    # Get mean test scores
    mean_scores = cv_results['mean_test_score']
    score_max = np.max(mean_scores)
    
    # Create multi-dimensional parameter grid
    param_dims = [len(values) for values in param_values.values()]
    total_combinations = np.prod(param_dims)
    
    _log(f"Parameter dimensions: {param_dims}")
    _log(f"Total combinations: {total_combinations}")
    _log(f"Actual results: {len(mean_scores)}")
    
    # Create multi-dimensional array for scores
    # We'll create a grid where each dimension corresponds to a parameter
    param_names_list = list(param_values.keys())
    param_values_list = list(param_values.values())
    
    # Create mapping from parameter values to indices
    param_value_to_idx = {}
    for i, (param_name, values) in enumerate(param_values.items()):
        param_value_to_idx[param_name] = {val: idx for idx, val in enumerate(values)}
    
    # Create multi-dimensional score array
    score_array = np.zeros(param_dims)
    score_array.fill(np.nan)  # Fill with NaN initially
    
    # Fill the array with actual scores
    best_idx = np.argmax(mean_scores)
    best_mapped = False
    
    for idx, score in enumerate(mean_scores):
        # Get parameter values for this combination
        param_vals = {}
        for param_name in param_names_list:
            param_vals[param_name] = cv_results[param_name][idx]
        
        # Convert classifier objects to names for comparison
        if 'param_classifier' in param_vals:
            param_vals['param_classifier'] = type(param_vals['param_classifier']).__name__
        
        # Find indices in the multi-dimensional array
        try:
            indices = []
            skip_this_combination = False
            for param_name in param_names_list:
                val = param_vals[param_name]
                if param_name == 'param_classifier':
                    val = type(cv_results[param_name][idx]).__name__
                
                # Handle MaskedConstant objects - these occur when a parameter doesn't apply to a classifier
                # For example, 'solver' is LogisticRegression-specific and will be masked for SVC
                if hasattr(val, '__class__') and 'MaskedConstant' in str(val.__class__):
                    # For classifier-specific parameters, we should still allow this combination
                    # but we need to skip this parameter dimension or use a default value
                    if idx == best_idx:
                        print(f"Note: Best combination has MaskedConstant for {param_name} (classifier-specific parameter)")
                    # Check if this parameter only has one value - if so, use index 0
                    if len(param_value_to_idx[param_name]) == 1:
                        idx_val = 0
                        indices.append(idx_val)
                    else:
                        # This parameter has multiple values but this combination doesn't use any of them
                        # We can't map this to the array properly
                        skip_this_combination = True
                        break
                    continue
                
                idx_val = param_value_to_idx[param_name][val]
                indices.append(idx_val)
            
            if not skip_this_combination:
                # Only set the score if we didn't skip this combination
                score_array[tuple(indices)] = score
                if idx == best_idx:
                    best_mapped = True
                    _log(f"Best score {score:.4f} mapped to array indices: {indices}")
        except (KeyError, ValueError) as e:
            if idx == best_idx:
                print(f"Error mapping best combination: {e}")
                print(f"  Parameter values: {param_vals}")
                print(f"  Available keys in mapping: {list(param_value_to_idx.keys())}")
            continue
    
    if not best_mapped:
        print("WARNING: Best parameter combination from grid search was not mapped to visualization array!")
        print(f"  This means the visualization will show slices for a different (lower) score.")
    
    # Use the original slices.py style visualization
    if feature_name is None:
        feature_name = "Hyperparameter_Grid_Search"
    result_path = output_dir
    
    # Debug: Check what maximum score is actually in the array
    actual_max_in_array = np.nanmax(score_array)
    _log(f"Max score from grid search: {score_max:.4f}")
    _log(f"Max score in visualization array: {actual_max_in_array:.4f}")
    
    if abs(score_max - actual_max_in_array) > 0.001:
        print(f"Warning: Max score mismatch! This might indicate that the best parameter combination")
        print(f"         couldn't be mapped to the visualization array (possibly due to classifier differences)")
        print(f"         Using the grid search max ({score_max:.4f}) for visualization.")
    
    # Create a single comprehensive visualization using slices.py style
    # Use score_max from grid search, not from the array
    saved_path = build_slices(
        score_array,
        param_values,
        score_max,
        feature_name,
        result_path,
        param_pairs=param_pairs,
        verbose=verbose
    )
    
    _log("Hyperparameter grid slices visualization completed!")
    if saved_path:
        _log(f"Saved comprehensive visualization to: {saved_path}")
    else:
        _log("Saved comprehensive visualization.")


def plot_score_dynamics(all_results: dict, output_path: str, invert_x: bool = True) -> None:
    """
    Plot score dynamics (accuracy, precision, recall, ROC-AUC) vs window position.

    Parameters
    ----------
    all_results : dict
        Dict with keys: window_positions, accuracy_mean, accuracy_std,
        precision_class_0_mean, precision_class_0_std, precision_class_1_mean,
        precision_class_1_std, recall_mean, recall_std, roc_auc_mean, roc_auc_std.
        Each value is a list of the same length as window_positions.
    output_path : str
        Path to save the figure (e.g. "Score_Dynamics.png").
    invert_x : bool, default=True
        If True, invert x-axis so position 4 appears on left, -8 on right.
    """
    plt.ioff()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Score Dynamics Across Window Positions', fontsize=16, fontweight='bold')

    window_positions_array = np.array(all_results['window_positions'])

    # Accuracy (mean only)
    ax = axes[0, 0]
    ax.plot(window_positions_array, all_results['accuracy_mean'], marker='o')
    ax.set_xlabel('Window Position')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Window Position')
    ax.grid(True, alpha=0.3)
    if invert_x:
        ax.invert_xaxis()

    # Precision Class 0 (mean only)
    ax = axes[0, 1]
    ax.plot(window_positions_array, all_results['precision_class_0_mean'], marker='o', color='green')
    ax.set_xlabel('Window Position')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Class 0 vs Window Position')
    ax.grid(True, alpha=0.3)
    if invert_x:
        ax.invert_xaxis()

    # Precision Class 1 (mean only)
    ax = axes[0, 2]
    ax.plot(window_positions_array, all_results['precision_class_1_mean'], marker='o', color='orange')
    ax.set_xlabel('Window Position')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Class 1 vs Window Position')
    ax.grid(True, alpha=0.3)
    if invert_x:
        ax.invert_xaxis()

    # Recall (mean only)
    ax = axes[1, 0]
    ax.plot(window_positions_array, all_results['recall_mean'], marker='o', color='red')
    ax.set_xlabel('Window Position')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Window Position')
    ax.grid(True, alpha=0.3)
    if invert_x:
        ax.invert_xaxis()

    # ROC-AUC (mean only)
    ax = axes[1, 1]
    ax.plot(window_positions_array, all_results['roc_auc_mean'], marker='o', color='purple')
    ax.set_xlabel('Window Position')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC vs Window Position')
    ax.grid(True, alpha=0.3)
    if invert_x:
        ax.invert_xaxis()

    # Remove empty subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    """
    Example usage of the hyperparameter grid visualizer.
    """
    print(__doc__)
    print("\nThis module is intended to be imported and used with sklearn GridSearchCV objects.")
    print("\nExample usage:")
    print("  from seismic_pipeline.visualization import visualize_hyperparameter_grid_slices")
    print("  # Visualize all parameter pairs:")
    print("  visualize_hyperparameter_grid_slices(grid_search, output_dir='./results')")
    print("  # Visualize only specific parameter pairs:")
    print("  visualize_hyperparameter_grid_slices(")
    print("      grid_search,")
    print("      output_dir='./results',")
    print("      param_pairs=[('param_C', 'param_gamma'), ('param_kernel', 'param_C')]")
    print("  )")


