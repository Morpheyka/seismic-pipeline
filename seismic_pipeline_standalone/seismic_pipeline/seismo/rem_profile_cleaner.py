"""
REM Profile Cleaner transformer.

This module provides a transformer that cleans REM profile data by replacing
zeros and NaN values with appropriate substitutes (like median values).
This helps improve data quality for downstream analysis.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..mod.sklearnbaseyt import TransformerMixinYt
from .logging_config import get_mod_logger


class REMProfileCleanerYt(TransformerMixinYt):
    """
    Cleans REM profile data by replacing zeros and NaN values.
    
    This transformer takes REM profile data and replaces problematic values
    (zeros, NaNs) with appropriate substitutes like median values or other
    statistical measures. This helps improve data quality for downstream analysis.
    """
    
    def __init__(self, 
                 replace_zeros: bool = True,
                 replace_nans: bool = True,
                 replacement_method: str = 'median',
                 replacement_value: Optional[float] = None,
                 min_valid_fraction: float = 0.5,
                 handle_empty_profiles: str = 'skip'):
        """
        Initialize the REM Profile Cleaner.
        
        Parameters
        ----------
        replace_zeros : bool, default=True
            Whether to replace zero values.
        replace_nans : bool, default=True
            Whether to replace NaN values.
        replacement_method : str, default='median'
            Method for calculating replacement values. Options: 'median', 'mean', 'mode', 'constant'.
        replacement_value : float, optional
            Constant value to use when replacement_method='constant'.
        min_valid_fraction : float, default=0.5
            Minimum fraction of valid (non-zero, non-NaN) values required to process a profile.
        handle_empty_profiles : str, default='skip'
            How to handle profiles with insufficient valid data. Options: 'skip', 'zero', 'nan'.
        """
        super().__init__()
        self.replace_zeros = replace_zeros
        self.replace_nans = replace_nans
        self.replacement_method = replacement_method
        self.replacement_value = replacement_value
        self.min_valid_fraction = min_valid_fraction
        self.handle_empty_profiles = handle_empty_profiles
        
        # Validate parameters
        valid_methods = ['median', 'mean', 'mode', 'constant']
        if replacement_method not in valid_methods:
            raise ValueError(f"Invalid replacement_method '{replacement_method}'. Valid options: {valid_methods}")
        
        if replacement_method == 'constant' and replacement_value is None:
            raise ValueError("replacement_value must be provided when replacement_method='constant'")
        
        valid_handlers = ['skip', 'zero', 'nan']
        if handle_empty_profiles not in valid_handlers:
            raise ValueError(f"Invalid handle_empty_profiles '{handle_empty_profiles}'. Valid options: {valid_handlers}")
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized REMProfileCleanerYt with method: {replacement_method}")
        
    def fit(self, X, y=None):
        """
        Fit the transformer by calculating replacement values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            REM profile data.
        y : array-like, optional
            Target values.
            
        Returns
        -------
        self : REMProfileCleanerYt
            Returns the instance itself.
        """
        self.logger.debug(f"Fitting REMProfileCleanerYt on {len(X) if hasattr(X, '__len__') else 'unknown'} samples")
        
        if not hasattr(X, '__len__') or len(X) == 0:
            self.replacement_value_ = 0.0
            return self
        
        # Collect all valid values for calculating replacement statistics
        all_valid_values = []
        
        for profile in X:
            if not isinstance(profile, np.ndarray):
                profile = np.array(profile)
            
            # Get valid values (non-zero, non-NaN)
            valid_mask = (profile != 0) & ~np.isnan(profile)
            valid_values = profile[valid_mask]
            
            if len(valid_values) > 0:
                all_valid_values.extend(valid_values)
        
        if len(all_valid_values) == 0:
            self.logger.warning("No valid values found in the data, using default replacement value")
            self.replacement_value_ = 0.0
        else:
            # Calculate replacement value based on method
            all_valid_values = np.array(all_valid_values)
            self.replacement_value_ = self._calculate_replacement_value(all_valid_values)
        
        self.logger.debug(f"Calculated replacement value: {self.replacement_value_}")
        
        return self
        
    def transform(self, X, y=None):
        """
        Transform REM profiles by cleaning zeros and NaNs.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            REM profile data.
        y : array-like, optional
            Target values.
            
        Returns
        -------
        X_transformed : array-like
            Cleaned REM profiles.
        y_transformed : array-like
            Target values (preserved).
        """
        self.logger.debug(f"Transforming {len(X) if hasattr(X, '__len__') else 'unknown'} REM profiles")
        
        if not hasattr(X, '__len__') or len(X) == 0:
            return X, y
        
        cleaned_profiles = []
        valid_indices = []
        
        for i, profile in enumerate(X):
            try:
                if not isinstance(profile, np.ndarray):
                    profile = np.array(profile)
                
                # Check if profile has sufficient valid data
                valid_mask = (profile != 0) & ~np.isnan(profile)
                valid_fraction = np.sum(valid_mask) / len(profile) if len(profile) > 0 else 0
                
                if valid_fraction < self.min_valid_fraction:
                    self.logger.debug(f"Profile {i} has insufficient valid data ({valid_fraction:.2f} < {self.min_valid_fraction})")
                    if self.handle_empty_profiles == 'skip':
                        continue
                    elif self.handle_empty_profiles == 'zero':
                        cleaned_profile = np.zeros_like(profile)
                    elif self.handle_empty_profiles == 'nan':
                        cleaned_profile = np.full_like(profile, np.nan)
                    else:
                        continue
                else:
                    # Clean the profile
                    cleaned_profile = self._clean_profile(profile)
                
                cleaned_profiles.append(cleaned_profile)
                valid_indices.append(i)
                
            except Exception as e:
                self.logger.warning(f"Error cleaning profile {i}: {e}")
                if self.handle_empty_profiles == 'skip':
                    continue
                else:
                    # Use replacement value for the entire profile
                    cleaned_profile = np.full_like(profile, self.replacement_value_)
                    cleaned_profiles.append(cleaned_profile)
                    valid_indices.append(i)
        
        if not cleaned_profiles:
            self.logger.warning("No profiles could be cleaned, returning empty result")
            return np.array([]).reshape(0, 0), y
        
        # Filter y if provided
        if y is not None and hasattr(y, '__getitem__'):
            if len(valid_indices) < len(y):
                if isinstance(y, (list, tuple)):
                    y_filtered = [y[i] for i in valid_indices]
                else:
                    y_filtered = y[valid_indices]
            else:
                y_filtered = y
        else:
            y_filtered = y
        
        self.logger.debug(f"Cleaned {len(cleaned_profiles)} profiles, removed {len(X) - len(cleaned_profiles)} invalid profiles")
        
        return cleaned_profiles, y_filtered
        
    def _calculate_replacement_value(self, valid_values: np.ndarray) -> float:
        """
        Calculate replacement value based on the specified method.
        
        Parameters
        ----------
        valid_values : np.ndarray
            Array of valid values to use for calculation.
            
        Returns
        -------
        float
            Calculated replacement value.
        """
        if self.replacement_method == 'median':
            return np.median(valid_values)
        elif self.replacement_method == 'mean':
            return np.mean(valid_values)
        elif self.replacement_method == 'mode':
            # Find the most frequent value
            unique_values, counts = np.unique(valid_values, return_counts=True)
            mode_idx = np.argmax(counts)
            return unique_values[mode_idx]
        elif self.replacement_method == 'constant':
            return self.replacement_value
        else:
            raise ValueError(f"Unknown replacement method: {self.replacement_method}")
    
    def _clean_profile(self, profile: np.ndarray) -> np.ndarray:
        """
        Clean a single REM profile by replacing zeros and NaNs.
        
        Parameters
        ----------
        profile : np.ndarray
            REM profile data.
            
        Returns
        -------
        np.ndarray
            Cleaned profile.
        """
        cleaned_profile = profile.copy()
        
        # Create mask for values to replace
        replace_mask = np.zeros_like(profile, dtype=bool)
        
        if self.replace_zeros:
            replace_mask |= (profile == 0)
        
        if self.replace_nans:
            replace_mask |= np.isnan(profile)
        
        # Replace masked values
        if np.any(replace_mask):
            cleaned_profile[replace_mask] = self.replacement_value_
            self.logger.debug(f"Replaced {np.sum(replace_mask)} values with {self.replacement_value_}")
        
        return cleaned_profile
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'replace_zeros': self.replace_zeros,
            'replace_nans': self.replace_nans,
            'replacement_method': self.replacement_method,
            'replacement_value': self.replacement_value,
            'min_valid_fraction': self.min_valid_fraction,
            'handle_empty_profiles': self.handle_empty_profiles
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


class REMProfileAdvancedCleanerYt(TransformerMixinYt):
    """
    Advanced REM profile cleaner with multiple cleaning strategies.
    
    This transformer provides more sophisticated cleaning strategies including
    interpolation, outlier detection, and adaptive replacement methods.
    """
    
    def __init__(self, 
                 cleaning_strategy: str = 'adaptive',
                 interpolation_method: str = 'linear',
                 outlier_threshold: float = 3.0,
                 min_valid_fraction: float = 0.3,
                 handle_empty_profiles: str = 'interpolate'):
        """
        Initialize the Advanced REM Profile Cleaner.
        
        Parameters
        ----------
        cleaning_strategy : str, default='adaptive'
            Cleaning strategy to use. Options: 'adaptive', 'interpolate', 'replace', 'robust'.
        interpolation_method : str, default='linear'
            Interpolation method for filling gaps. Options: 'linear', 'cubic', 'nearest'.
        outlier_threshold : float, default=3.0
            Threshold for outlier detection (in standard deviations).
        min_valid_fraction : float, default=0.3
            Minimum fraction of valid values required.
        handle_empty_profiles : str, default='interpolate'
            How to handle empty profiles. Options: 'interpolate', 'skip', 'zero', 'nan'.
        """
        super().__init__()
        self.cleaning_strategy = cleaning_strategy
        self.interpolation_method = interpolation_method
        self.outlier_threshold = outlier_threshold
        self.min_valid_fraction = min_valid_fraction
        self.handle_empty_profiles = handle_empty_profiles
        
        # Validate parameters
        valid_strategies = ['adaptive', 'interpolate', 'replace', 'robust']
        if cleaning_strategy not in valid_strategies:
            raise ValueError(f"Invalid cleaning_strategy '{cleaning_strategy}'. Valid options: {valid_strategies}")
        
        valid_interp = ['linear', 'cubic', 'nearest']
        if interpolation_method not in valid_interp:
            raise ValueError(f"Invalid interpolation_method '{interpolation_method}'. Valid options: {valid_interp}")
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized REMProfileAdvancedCleanerYt with strategy: {cleaning_strategy}")
        
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.logger.debug(f"Fitting REMProfileAdvancedCleanerYt on {len(X) if hasattr(X, '__len__') else 'unknown'} samples")
        
        # Calculate global statistics for adaptive cleaning
        if self.cleaning_strategy == 'adaptive':
            self._calculate_global_statistics(X)
        
        return self
        
    def transform(self, X, y=None):
        """Transform REM profiles using advanced cleaning strategies."""
        if not hasattr(X, '__len__') or len(X) == 0:
            return X, y
        
        cleaned_profiles = []
        valid_indices = []
        
        for i, profile in enumerate(X):
            try:
                if not isinstance(profile, np.ndarray):
                    profile = np.array(profile)
                
                # Check data quality
                valid_mask = ~np.isnan(profile) & (profile != 0)
                valid_fraction = np.sum(valid_mask) / len(profile) if len(profile) > 0 else 0
                
                if valid_fraction < self.min_valid_fraction:
                    if self.handle_empty_profiles == 'skip':
                        continue
                    elif self.handle_empty_profiles == 'interpolate':
                        cleaned_profile = self._interpolate_profile(profile)
                    elif self.handle_empty_profiles == 'zero':
                        cleaned_profile = np.zeros_like(profile)
                    elif self.handle_empty_profiles == 'nan':
                        cleaned_profile = np.full_like(profile, np.nan)
                    else:
                        continue
                else:
                    # Apply cleaning strategy
                    cleaned_profile = self._apply_cleaning_strategy(profile)
                
                cleaned_profiles.append(cleaned_profile)
                valid_indices.append(i)
                
            except Exception as e:
                self.logger.warning(f"Error cleaning profile {i}: {e}")
                continue
        
        if not cleaned_profiles:
            return np.array([]).reshape(0, 0), y
        
        # Filter y if provided
        if y is not None and hasattr(y, '__getitem__'):
            if len(valid_indices) < len(y):
                if isinstance(y, (list, tuple)):
                    y_filtered = [y[i] for i in valid_indices]
                else:
                    y_filtered = y[valid_indices]
            else:
                y_filtered = y
        else:
            y_filtered = y
        
        return cleaned_profiles, y_filtered
        
    def _calculate_global_statistics(self, X):
        """Calculate global statistics for adaptive cleaning."""
        all_values = []
        for profile in X:
            if isinstance(profile, np.ndarray):
                valid_values = profile[~np.isnan(profile) & (profile != 0)]
                all_values.extend(valid_values)
        
        if all_values:
            all_values = np.array(all_values)
            self.global_median_ = np.median(all_values)
            self.global_std_ = np.std(all_values)
            self.global_mean_ = np.mean(all_values)
        else:
            self.global_median_ = 0.0
            self.global_std_ = 1.0
            self.global_mean_ = 0.0
        
        self.logger.debug(f"Global stats - median: {self.global_median_}, std: {self.global_std_}")
        
    def _apply_cleaning_strategy(self, profile: np.ndarray) -> np.ndarray:
        """Apply the specified cleaning strategy to a profile."""
        if self.cleaning_strategy == 'adaptive':
            return self._adaptive_clean(profile)
        elif self.cleaning_strategy == 'interpolate':
            return self._interpolate_profile(profile)
        elif self.cleaning_strategy == 'replace':
            return self._replace_clean(profile)
        elif self.cleaning_strategy == 'robust':
            return self._robust_clean(profile)
        else:
            return profile
    
    def _adaptive_clean(self, profile: np.ndarray) -> np.ndarray:
        """Adaptive cleaning based on local and global statistics."""
        cleaned = profile.copy()
        
        # Replace zeros and NaNs with local median
        invalid_mask = (profile == 0) | np.isnan(profile)
        if np.any(invalid_mask):
            valid_values = profile[~invalid_mask]
            if len(valid_values) > 0:
                local_median = np.median(valid_values)
                cleaned[invalid_mask] = local_median
            else:
                cleaned[invalid_mask] = self.global_median_
        
        # Remove outliers
        if len(cleaned) > 3:
            z_scores = np.abs((cleaned - np.mean(cleaned)) / (np.std(cleaned) + 1e-8))
            outlier_mask = z_scores > self.outlier_threshold
            if np.any(outlier_mask):
                cleaned[outlier_mask] = np.median(cleaned[~outlier_mask])
        
        return cleaned
    
    def _interpolate_profile(self, profile: np.ndarray) -> np.ndarray:
        """Interpolate missing values in the profile."""
        from scipy import interpolate
        
        cleaned = profile.copy()
        valid_mask = ~np.isnan(profile) & (profile != 0)
        
        if np.sum(valid_mask) < 2:
            # Not enough points for interpolation, use median
            if hasattr(self, 'global_median_'):
                cleaned[~valid_mask] = self.global_median_
            else:
                cleaned[~valid_mask] = 0.0
            return cleaned
        
        try:
            valid_indices = np.where(valid_mask)[0]
            valid_values = profile[valid_indices]
            
            # Create interpolation function
            if self.interpolation_method == 'linear':
                f = interpolate.interp1d(valid_indices, valid_values, 
                                       kind='linear', bounds_error=False, 
                                       fill_value='extrapolate')
            elif self.interpolation_method == 'cubic':
                f = interpolate.interp1d(valid_indices, valid_values, 
                                       kind='cubic', bounds_error=False, 
                                       fill_value='extrapolate')
            else:  # nearest
                f = interpolate.interp1d(valid_indices, valid_values, 
                                       kind='nearest', bounds_error=False, 
                                       fill_value='extrapolate')
            
            # Interpolate missing values
            all_indices = np.arange(len(profile))
            interpolated = f(all_indices)
            cleaned[~valid_mask] = interpolated[~valid_mask]
            
        except Exception as e:
            self.logger.warning(f"Interpolation failed: {e}, using median replacement")
            if hasattr(self, 'global_median_'):
                cleaned[~valid_mask] = self.global_median_
            else:
                cleaned[~valid_mask] = 0.0
        
        return cleaned
    
    def _replace_clean(self, profile: np.ndarray) -> np.ndarray:
        """Simple replacement cleaning."""
        cleaned = profile.copy()
        invalid_mask = (profile == 0) | np.isnan(profile)
        
        if np.any(invalid_mask):
            valid_values = profile[~invalid_mask]
            if len(valid_values) > 0:
                replacement = np.median(valid_values)
            else:
                replacement = getattr(self, 'global_median_', 0.0)
            cleaned[invalid_mask] = replacement
        
        return cleaned
    
    def _robust_clean(self, profile: np.ndarray) -> np.ndarray:
        """Robust cleaning using robust statistics."""
        cleaned = profile.copy()
        
        # Replace invalid values
        invalid_mask = (profile == 0) | np.isnan(profile)
        if np.any(invalid_mask):
            valid_values = profile[~invalid_mask]
            if len(valid_values) > 0:
                # Use median as it's more robust than mean
                replacement = np.median(valid_values)
                cleaned[invalid_mask] = replacement
            else:
                cleaned[invalid_mask] = getattr(self, 'global_median_', 0.0)
        
        # Remove outliers using IQR method
        if len(cleaned) > 4:
            q25, q75 = np.percentile(cleaned, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_mask = (cleaned < lower_bound) | (cleaned > upper_bound)
            if np.any(outlier_mask):
                cleaned[outlier_mask] = np.median(cleaned[~outlier_mask])
        
        return cleaned
