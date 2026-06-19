"""
Daily REM Profile Extractor - Extract per-day statistics from REM profiles.

This module provides transformers to extract daily statistics from REM profiles,
such as max-min difference for each day in the window.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from ..mod.sklearnbaseyt import TransformerMixinYt
from .logging_config import get_mod_logger


class REMDailyExtractorYt(TransformerMixinYt):
    """
    Extract daily statistics from REM profiles.
    
    This transformer takes REM profile data and extracts statistics for each day
    in the window, such as max-min difference, mean, etc. The number of output
    features equals the number of days in the window.
    
    For example, if you have a 3-day window, you get 3 features (one per day).
    """
    
    def __init__(self, 
                 daily_statistic: str = 'max_min_diff',
                 window_days: int = 3,
                 handle_empty_days: str = 'zero'):
        """
        Initialize the Daily REM Profile Extractor.
        
        Parameters
        ----------
        daily_statistic : str, default='max_min_diff'
            Statistic to extract for each day. Options:
            - 'max_min_diff': max - min for each day
            - 'max': maximum value for each day
            - 'min': minimum value for each day
            - 'mean': mean value for each day
            - 'std': standard deviation for each day
            - 'range': range (max - min) for each day
            - 'sum': sum of all values for each day
        window_days : int, default=3
            Number of days in the window (determines number of output features)
        handle_empty_days : str, default='zero'
            How to handle days with no REM data. Options: 'zero', 'nan', 'skip'
        """
        super().__init__()
        self.daily_statistic = daily_statistic
        self.window_days = window_days
        self.handle_empty_days = handle_empty_days
        
        # Validate daily statistic
        valid_stats = ['max_min_diff', 'max', 'min', 'mean', 'std', 'range', 'sum']
        if daily_statistic not in valid_stats:
            raise ValueError(f"Invalid daily_statistic '{daily_statistic}'. Valid options: {valid_stats}")
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized REMDailyExtractorYt with statistic: {daily_statistic}, window_days: {window_days}")
        
    def fit(self, X, y=None):
        """
        Fit the transformer.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            REM profile data.
        y : array-like, optional
            Target values.
            
        Returns
        -------
        self : REMDailyExtractorYt
            Returns the instance itself.
        """
        # Store feature names for later use
        self.feature_names_ = [f'day_{i+1}_{self.daily_statistic}' for i in range(self.window_days)]
        
        self.logger.debug(f"Fitted REMDailyExtractorYt with {len(X)} samples")
        return self
        
    def transform(self, X, y=None):
        """
        Transform REM profiles by extracting daily statistics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            REM profile data.
        y : array-like, optional
            Target values.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, window_days)
            Daily statistics extracted from REM profiles.
        y_transformed : array-like
            Target values (preserved).
        """
        if len(X) == 0:
            return np.array([]).reshape(0, self.window_days), y
        
        extracted_stats = []
        
        for i, profile in enumerate(X):
            try:
                daily_stats = self._extract_daily_statistics(profile)
                extracted_stats.append(daily_stats)
                self.logger.debug(f"Extracted daily stats for sample {i}: {daily_stats}")
            except Exception as e:
                self.logger.warning(f"Failed to extract daily stats for sample {i}: {e}")
                # Handle error case
                if self.handle_empty_days == 'zero':
                    daily_stats = np.zeros(self.window_days)
                elif self.handle_empty_days == 'nan':
                    daily_stats = np.full(self.window_days, np.nan)
                else:  # skip
                    continue
                extracted_stats.append(daily_stats)
        
        X_transformed = np.array(extracted_stats)
        
        self.logger.info(f"Transformed {len(X)} samples to {X_transformed.shape}")
        
        return X_transformed, y
        
    def _extract_daily_statistics(self, profile: np.ndarray) -> np.ndarray:
        """
        Extract daily statistics from a single REM profile.
        
        This method assumes that the profile contains REM percentages for multiple days
        concatenated together. It splits the profile by day and calculates the requested
        statistic for each day.
        
        Parameters
        ----------
        profile : np.ndarray
            REM profile data (array of REM percentages for multiple days).
            
        Returns
        -------
        np.ndarray
            Array of daily statistics (one per day).
        """
        if len(profile) == 0:
            if self.handle_empty_days == 'zero':
                return np.zeros(self.window_days)
            elif self.handle_empty_days == 'nan':
                return np.full(self.window_days, np.nan)
            else:
                return np.array([])
        
        # Estimate samples per day based on total length and window_days
        # This is a heuristic - in practice, you might want to pass this information
        samples_per_day = len(profile) // self.window_days
        
        if samples_per_day == 0:
            # Not enough data points
            if self.handle_empty_days == 'zero':
                return np.zeros(self.window_days)
            elif self.handle_empty_days == 'nan':
                return np.full(self.window_days, np.nan)
            else:
                return np.array([])
        
        daily_stats = []
        
        for day_idx in range(self.window_days):
            start_idx = day_idx * samples_per_day
            end_idx = start_idx + samples_per_day
            
            # Handle last day which might have different length
            if day_idx == self.window_days - 1:
                end_idx = len(profile)
            
            day_profile = profile[start_idx:end_idx]
            
            if len(day_profile) == 0:
                # Empty day
                if self.handle_empty_days == 'zero':
                    stat_value = 0.0
                elif self.handle_empty_days == 'nan':
                    stat_value = np.nan
                else:  # skip
                    continue
            else:
                # Calculate the requested statistic for this day
                stat_value = self._calculate_daily_statistic(day_profile)
            
            daily_stats.append(stat_value)
        
        return np.array(daily_stats)
    
    def _calculate_daily_statistic(self, day_profile: np.ndarray) -> float:
        """
        Calculate the requested statistic for a single day's profile.
        
        Parameters
        ----------
        day_profile : np.ndarray
            REM profile for a single day.
            
        Returns
        -------
        float
            Calculated statistic value.
        """
        if len(day_profile) == 0:
            return 0.0
        
        if self.daily_statistic == 'max_min_diff':
            return np.max(day_profile) - np.min(day_profile)
        elif self.daily_statistic == 'max':
            return np.max(day_profile)
        elif self.daily_statistic == 'min':
            return np.min(day_profile)
        elif self.daily_statistic == 'mean':
            return np.mean(day_profile)
        elif self.daily_statistic == 'std':
            return np.std(day_profile)
        elif self.daily_statistic == 'range':
            return np.max(day_profile) - np.min(day_profile)
        elif self.daily_statistic == 'sum':
            return np.sum(day_profile)
        else:
            raise ValueError(f"Unknown daily statistic: {self.daily_statistic}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the extracted daily statistics.
        
        Returns
        -------
        list of str
            Feature names.
        """
        return self.feature_names_.copy()
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'daily_statistic': self.daily_statistic,
            'window_days': self.window_days,
            'handle_empty_days': self.handle_empty_days
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self


class REMDailyMultiStatExtractorYt(TransformerMixinYt):
    """
    Extract multiple daily statistics from REM profiles.
    
    This transformer extracts multiple statistics for each day in the window.
    For example, with 3 days and 2 statistics, you get 6 features total.
    """
    
    def __init__(self, 
                 daily_statistics: List[str] = None,
                 window_days: int = 3,
                 handle_empty_days: str = 'zero'):
        """
        Initialize the Multi-Stat Daily REM Profile Extractor.
        
        Parameters
        ----------
        daily_statistics : list of str, default=['max_min_diff', 'mean']
            List of statistics to extract for each day.
        window_days : int, default=3
            Number of days in the window.
        handle_empty_days : str, default='zero'
            How to handle days with no REM data.
        """
        super().__init__()
        self.daily_statistics = daily_statistics or ['max_min_diff', 'mean']
        self.window_days = window_days
        self.handle_empty_days = handle_empty_days
        
        # Validate statistics
        valid_stats = ['max_min_diff', 'max', 'min', 'mean', 'std', 'range', 'sum']
        for stat in self.daily_statistics:
            if stat not in valid_stats:
                raise ValueError(f"Invalid statistic '{stat}'. Valid options: {valid_stats}")
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized REMDailyMultiStatExtractorYt with stats: {daily_statistics}, window_days: {window_days}")
        
    def fit(self, X, y=None):
        """Fit the transformer."""
        # Store feature names for later use
        self.feature_names_ = []
        for day_idx in range(self.window_days):
            for stat in self.daily_statistics:
                self.feature_names_.append(f'day_{day_idx+1}_{stat}')
        
        self.logger.debug(f"Fitted REMDailyMultiStatExtractorYt with {len(X)} samples")
        return self
        
    def transform(self, X, y=None):
        """Transform REM profiles by extracting multiple daily statistics."""
        if len(X) == 0:
            expected_features = self.window_days * len(self.daily_statistics)
            return np.array([]).reshape(0, expected_features), y
        
        extracted_stats = []
        
        for i, profile in enumerate(X):
            try:
                daily_stats = self._extract_multi_daily_statistics(profile)
                extracted_stats.append(daily_stats)
                self.logger.debug(f"Extracted multi daily stats for sample {i}: {daily_stats}")
            except Exception as e:
                self.logger.warning(f"Failed to extract multi daily stats for sample {i}: {e}")
                # Handle error case
                expected_features = self.window_days * len(self.daily_statistics)
                if self.handle_empty_days == 'zero':
                    daily_stats = np.zeros(expected_features)
                elif self.handle_empty_days == 'nan':
                    daily_stats = np.full(expected_features, np.nan)
                else:  # skip
                    continue
                extracted_stats.append(daily_stats)
        
        X_transformed = np.array(extracted_stats)
        
        self.logger.info(f"Transformed {len(X)} samples to {X_transformed.shape}")
        
        return X_transformed, y
        
    def _extract_multi_daily_statistics(self, profile: np.ndarray) -> np.ndarray:
        """Extract multiple daily statistics from a single REM profile."""
        if len(profile) == 0:
            expected_features = self.window_days * len(self.daily_statistics)
            if self.handle_empty_days == 'zero':
                return np.zeros(expected_features)
            elif self.handle_empty_days == 'nan':
                return np.full(expected_features, np.nan)
            else:
                return np.array([])
        
        # Estimate samples per day
        samples_per_day = len(profile) // self.window_days
        
        if samples_per_day == 0:
            expected_features = self.window_days * len(self.daily_statistics)
            if self.handle_empty_days == 'zero':
                return np.zeros(expected_features)
            elif self.handle_empty_days == 'nan':
                return np.full(expected_features, np.nan)
            else:
                return np.array([])
        
        all_daily_stats = []
        
        for day_idx in range(self.window_days):
            start_idx = day_idx * samples_per_day
            end_idx = start_idx + samples_per_day
            
            # Handle last day
            if day_idx == self.window_days - 1:
                end_idx = len(profile)
            
            day_profile = profile[start_idx:end_idx]
            
            if len(day_profile) == 0:
                # Empty day
                if self.handle_empty_days == 'zero':
                    day_stats = [0.0] * len(self.daily_statistics)
                elif self.handle_empty_days == 'nan':
                    day_stats = [np.nan] * len(self.daily_statistics)
                else:  # skip
                    continue
            else:
                # Calculate all requested statistics for this day
                day_stats = []
                for stat in self.daily_statistics:
                    stat_value = self._calculate_daily_statistic(day_profile, stat)
                    day_stats.append(stat_value)
            
            all_daily_stats.extend(day_stats)
        
        return np.array(all_daily_stats)
    
    def _calculate_daily_statistic(self, day_profile: np.ndarray, statistic: str) -> float:
        """Calculate a specific statistic for a single day's profile."""
        if len(day_profile) == 0:
            return 0.0
        
        if statistic == 'max_min_diff':
            return np.max(day_profile) - np.min(day_profile)
        elif statistic == 'max':
            return np.max(day_profile)
        elif statistic == 'min':
            return np.min(day_profile)
        elif statistic == 'mean':
            return np.mean(day_profile)
        elif statistic == 'std':
            return np.std(day_profile)
        elif statistic == 'range':
            return np.max(day_profile) - np.min(day_profile)
        elif statistic == 'sum':
            return np.sum(day_profile)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for the extracted daily statistics."""
        return self.feature_names_.copy()
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'daily_statistics': self.daily_statistics,
            'window_days': self.window_days,
            'handle_empty_days': self.handle_empty_days
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self



class REMDailyMultiDynamicStatExtractorYt(TransformerMixinYt):
    """
    Extract multiple dynamic daily statistics from REM profiles.
    
    This transformer extracts multiple statistics for each day in the window.
    For example, with 3 days and 2 statistics, you get 6 features total.
    """
    
    def __init__(self, 
                 daily_statistics: List[str] = None,
                 window_days: int = 3,
                 handle_empty_days: str = 'zero'):
        """
        Initialize the Multi-Stat Daily REM Profile Extractor.
        
        Parameters
        ----------
        daily_statistics : list of str, default=['max_min_diff', 'mean']
            List of statistics to extract for each day.
        window_days : int, default=3
            Number of days in the window.
        handle_empty_days : str, default='zero'
            How to handle days with no REM data.
        """
        super().__init__()
        self.daily_statistics = daily_statistics or ['max_min_diff', 'mean']
        self.window_days = window_days
        self.handle_empty_days = handle_empty_days
        
        # Validate statistics
        valid_stats = ['max_min_diff', 'max', 'min', 'mean', 'std', 'range', 'sum']
        for stat in self.daily_statistics:
            if stat not in valid_stats:
                raise ValueError(f"Invalid statistic '{stat}'. Valid options: {valid_stats}")
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized REMDailyMultiStatExtractorYt with stats: {daily_statistics}, window_days: {window_days}")
        
    def fit(self, X, y=None):
        """Fit the transformer."""
        # Store feature names for later use
        self.feature_names_ = []
        for day_idx in range(self.window_days):
            for stat in self.daily_statistics:
                self.feature_names_.append(f'day_{day_idx+1}_{stat}')
        
        self.logger.debug(f"Fitted REMDailyMultiStatExtractorYt with {len(X)} samples")
        return self
        
    def transform(self, X, y=None):
        """Transform REM profiles by extracting multiple daily statistics."""
        if len(X) == 0:
            expected_features = self.window_days * len(self.daily_statistics)
            return np.array([]).reshape(0, expected_features), y
        
        extracted_stats = []
        
        for i, profile in enumerate(X):
            try:
                daily_stats = self._extract_multi_daily_statistics(profile)
                extracted_stats.append(daily_stats)
                self.logger.debug(f"Extracted multi daily stats for sample {i}: {daily_stats}")
            except Exception as e:
                self.logger.warning(f"Failed to extract multi daily stats for sample {i}: {e}")
                # Handle error case
                expected_features = self.window_days * len(self.daily_statistics)
                if self.handle_empty_days == 'zero':
                    daily_stats = np.zeros(expected_features)
                elif self.handle_empty_days == 'nan':
                    daily_stats = np.full(expected_features, np.nan)
                else:  # skip
                    continue
                extracted_stats.append(daily_stats)
        
        X_transformed = np.array(extracted_stats)
        
        self.logger.info(f"Transformed {len(X)} samples to {X_transformed.shape}")
        
        return X_transformed, y
        
    def _extract_multi_daily_statistics(self, profile: np.ndarray) -> np.ndarray:
        """Extract multiple daily statistics from a single REM profile."""
        if len(profile) == 0:
            expected_features = self.window_days * len(self.daily_statistics)
            if self.handle_empty_days == 'zero':
                return np.zeros(expected_features)
            elif self.handle_empty_days == 'nan':
                return np.full(expected_features, np.nan)
            else:
                return np.array([])
        
        # Estimate samples per day
        samples_per_day = len(profile) // self.window_days
        
        if samples_per_day == 0:
            expected_features = self.window_days * len(self.daily_statistics)
            if self.handle_empty_days == 'zero':
                return np.zeros(expected_features)
            elif self.handle_empty_days == 'nan':
                return np.full(expected_features, np.nan)
            else:
                return np.array([])
        
        all_daily_stats = []
        
        for day_idx in range(self.window_days):
            start_idx = day_idx * samples_per_day
            end_idx = start_idx + samples_per_day
            
            # Handle last day
            if day_idx == self.window_days - 1:
                end_idx = len(profile)
            
            day_profile = profile[start_idx:end_idx]
            
            if len(day_profile) == 0:
                # Empty day
                if self.handle_empty_days == 'zero':
                    day_stats = [0.0] * len(self.daily_statistics)
                elif self.handle_empty_days == 'nan':
                    day_stats = [np.nan] * len(self.daily_statistics)
                else:  # skip
                    continue
            else:
                # Calculate all requested statistics for this day
                day_stats = []
                for stat in self.daily_statistics:
                    stat_value = self._calculate_daily_statistic(day_profile, stat)
                    day_stats.append(stat_value)
            
            all_daily_stats.extend(day_stats)
        
        return np.array(all_daily_stats)
    
    def _calculate_daily_statistic(self, day_profile: np.ndarray, statistic: str) -> float:
        """Calculate a specific statistic for a single day's profile."""
        if len(day_profile) == 0:
            return 0.0
        
        if statistic == 'max_min_diff':
            return np.max(day_profile) - np.min(day_profile)
        elif statistic == 'max':
            return np.max(day_profile)
        elif statistic == 'min':
            return np.min(day_profile)
        elif statistic == 'mean':
            return np.mean(day_profile)
        elif statistic == 'std':
            return np.std(day_profile)
        elif statistic == 'range':
            return np.max(day_profile) - np.min(day_profile)
        elif statistic == 'sum':
            return np.sum(day_profile)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for the extracted daily statistics."""
        return self.feature_names_.copy()
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'daily_statistics': self.daily_statistics,
            'window_days': self.window_days,
            'handle_empty_days': self.handle_empty_days
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self


