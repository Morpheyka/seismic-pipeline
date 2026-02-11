"""
REM Profile Max-Min Extractor transformer.

This module provides a transformer that extracts maximum and minimum values
from REM profiles, replacing the full profile data with these summary statistics.
This is useful for dimensionality reduction and feature extraction.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..mod.sklearnbaseyt import TransformerMixinYt
from .logging_config import get_mod_logger


class REMProfileMaxMinExtractorYt(TransformerMixinYt):
    """
    Extracts maximum and minimum values from REM profiles.
    
    This transformer takes REM profile data (arrays of REM percentages over time)
    and extracts the maximum and minimum values, replacing the full profile
    with these summary statistics. This is useful for dimensionality reduction
    while preserving key information about REM sleep patterns.
    """
    
    def __init__(self, 
                 include_other_stats: bool = False,
                 stats_to_extract: Optional[List[str]] = None,
                 handle_empty_profiles: str = 'zero'):
        """
        Initialize the REM Profile Max-Min Extractor.
        
        Parameters
        ----------
        include_other_stats : bool, default=False
            Whether to include additional statistics beyond max/min.
        stats_to_extract : list of str, optional
            List of statistics to extract. If None, uses ['max', 'min'].
            Available options: 'max', 'min', 'mean', 'std', 'median', 'range', 'q25', 'q75'.
        handle_empty_profiles : str, default='zero'
            How to handle empty profiles. Options: 'zero', 'nan', 'skip'.
        """
        super().__init__()
        self.include_other_stats = include_other_stats
        self.handle_empty_profiles = handle_empty_profiles
        
        if stats_to_extract is None:
            if include_other_stats:
                self.stats_to_extract = ['max', 'min', 'mean', 'std', 'range']
            else:
                self.stats_to_extract = ['max', 'min']
        else:
            self.stats_to_extract = stats_to_extract
            
        # Validate stats
        valid_stats = ['max', 'min', 'mean', 'std', 'median', 'range', 'q25', 'q75']
        for stat in self.stats_to_extract:
            if stat not in valid_stats:
                raise ValueError(f"Invalid statistic '{stat}'. Valid options: {valid_stats}")
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized REMProfileMaxMinExtractorYt with stats: {self.stats_to_extract}")
        
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
        self : REMProfileMaxMinExtractorYt
            Returns the instance itself.
        """
        self.logger.debug(f"Fitting REMProfileMaxMinExtractorYt on {len(X) if hasattr(X, '__len__') else 'unknown'} samples")
        
        # Determine the number of output features
        self.n_output_features_ = len(self.stats_to_extract)
        
        # Store feature names for reference
        self.feature_names_ = [f"rem_profile_{stat}" for stat in self.stats_to_extract]
        
        self.logger.debug(f"Will extract {self.n_output_features_} features: {self.feature_names_}")
        
        return self
        
    def transform(self, X, y=None):
        """
        Transform REM profiles by extracting max-min and other statistics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            REM profile data.
        y : array-like, optional
            Target values.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_output_features)
            Extracted statistics from REM profiles.
        y_transformed : array-like
            Target values (preserved).
        """
        self.logger.debug(f"Transforming {len(X) if hasattr(X, '__len__') else 'unknown'} REM profiles")
        
        if not hasattr(X, '__len__') or len(X) == 0:
            return np.array([]).reshape(0, self.n_output_features_), y
        
        extracted_stats = []
        
        for i, profile in enumerate(X):
            try:
                # Convert to numpy array if needed
                if not isinstance(profile, np.ndarray):
                    profile = np.array(profile)
                
                # Handle empty profiles
                if len(profile) == 0 or np.all(np.isnan(profile)):
                    if self.handle_empty_profiles == 'zero':
                        stats = np.zeros(self.n_output_features_)
                    elif self.handle_empty_profiles == 'nan':
                        stats = np.full(self.n_output_features_, np.nan)
                    elif self.handle_empty_profiles == 'skip':
                        continue
                    else:
                        raise ValueError(f"Unknown handle_empty_profiles option: {self.handle_empty_profiles}")
                else:
                    # Extract statistics
                    stats = self._extract_statistics(profile)
                
                extracted_stats.append(stats)
                
            except Exception as e:
                self.logger.warning(f"Error processing profile {i}: {e}")
                if self.handle_empty_profiles == 'zero':
                    stats = np.zeros(self.n_output_features_)
                elif self.handle_empty_profiles == 'nan':
                    stats = np.full(self.n_output_features_, np.nan)
                else:
                    continue
                extracted_stats.append(stats)
        
        if not extracted_stats:
            return np.array([]).reshape(0, self.n_output_features_), y
            
        X_transformed = np.array(extracted_stats)
        
        self.logger.debug(f"Extracted statistics shape: {X_transformed.shape}")
        
        return X_transformed, y
        
    def _extract_statistics(self, profile: np.ndarray) -> np.ndarray:
        """
        Extract statistics from a single REM profile.
        
        Parameters
        ----------
        profile : np.ndarray
            REM profile data (array of REM percentages).
            
        Returns
        -------
        np.ndarray
            Array of extracted statistics.
        """
        stats = []
        
        for stat in self.stats_to_extract:
            if stat == 'max':
                value = np.max(profile)
            elif stat == 'min':
                value = np.min(profile)
            elif stat == 'mean':
                value = np.mean(profile)
            elif stat == 'std':
                value = np.std(profile)
            elif stat == 'median':
                value = np.median(profile)
            elif stat == 'range':
                value = np.max(profile) - np.min(profile)
            elif stat == 'q25':
                value = np.percentile(profile, 25)
            elif stat == 'q75':
                value = np.percentile(profile, 75)
            else:
                raise ValueError(f"Unknown statistic: {stat}")
            
            stats.append(value)
        
        return np.array(stats)
        
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the extracted statistics.
        
        Returns
        -------
        list of str
            Feature names.
        """
        return self.feature_names_.copy()
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'include_other_stats': self.include_other_stats,
            'stats_to_extract': self.stats_to_extract,
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


class REMProfileSummaryExtractorYt(TransformerMixinYt):
    """
    Extracts comprehensive summary statistics from REM profiles.
    
    This transformer provides a more comprehensive set of statistics
    for REM profile analysis, including temporal patterns and distribution
    characteristics.
    """
    
    def __init__(self, 
                 include_temporal_stats: bool = True,
                 include_distribution_stats: bool = True,
                 include_peak_stats: bool = True,
                 handle_empty_profiles: str = 'zero'):
        """
        Initialize the REM Profile Summary Extractor.
        
        Parameters
        ----------
        include_temporal_stats : bool, default=True
            Whether to include temporal statistics (trend, slope, etc.).
        include_distribution_stats : bool, default=True
            Whether to include distribution statistics (skewness, kurtosis, etc.).
        include_peak_stats : bool, default=True
            Whether to include peak-related statistics.
        handle_empty_profiles : str, default='zero'
            How to handle empty profiles.
        """
        super().__init__()
        self.include_temporal_stats = include_temporal_stats
        self.include_distribution_stats = include_distribution_stats
        self.include_peak_stats = include_peak_stats
        self.handle_empty_profiles = handle_empty_profiles
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.info("Initialized REMProfileSummaryExtractorYt")
        
    def fit(self, X, y=None):
        """Fit the transformer."""
        # Build feature names based on enabled statistics
        self.feature_names_ = []
        
        # Basic statistics
        self.feature_names_.extend(['rem_max', 'rem_min', 'rem_mean', 'rem_std', 'rem_range'])
        
        if self.include_temporal_stats:
            self.feature_names_.extend(['rem_trend', 'rem_slope', 'rem_variance'])
            
        if self.include_distribution_stats:
            self.feature_names_.extend(['rem_skewness', 'rem_kurtosis', 'rem_q25', 'rem_q75'])
            
        if self.include_peak_stats:
            self.feature_names_.extend(['rem_peak_count', 'rem_peak_mean', 'rem_peak_std'])
        
        self.n_output_features_ = len(self.feature_names_)
        self.logger.debug(f"Will extract {self.n_output_features_} features: {self.feature_names_}")
        
        return self
        
    def transform(self, X, y=None):
        """Transform REM profiles by extracting comprehensive statistics."""
        if not hasattr(X, '__len__') or len(X) == 0:
            return np.array([]).reshape(0, self.n_output_features_), y
        
        extracted_stats = []
        
        for i, profile in enumerate(X):
            try:
                if not isinstance(profile, np.ndarray):
                    profile = np.array(profile)
                
                if len(profile) == 0 or np.all(np.isnan(profile)):
                    if self.handle_empty_profiles == 'zero':
                        stats = np.zeros(self.n_output_features_)
                    elif self.handle_empty_profiles == 'nan':
                        stats = np.full(self.n_output_features_, np.nan)
                    else:
                        continue
                else:
                    stats = self._extract_comprehensive_statistics(profile)
                
                extracted_stats.append(stats)
                
            except Exception as e:
                self.logger.warning(f"Error processing profile {i}: {e}")
                if self.handle_empty_profiles == 'zero':
                    stats = np.zeros(self.n_output_features_)
                elif self.handle_empty_profiles == 'nan':
                    stats = np.full(self.n_output_features_, np.nan)
                else:
                    continue
                extracted_stats.append(stats)
        
        if not extracted_stats:
            return np.array([]).reshape(0, self.n_output_features_), y
            
        return np.array(extracted_stats), y
        
    def _extract_comprehensive_statistics(self, profile: np.ndarray) -> np.ndarray:
        """Extract comprehensive statistics from a REM profile."""
        stats = []
        
        # Basic statistics
        stats.extend([
            np.max(profile),
            np.min(profile),
            np.mean(profile),
            np.std(profile),
            np.max(profile) - np.min(profile)
        ])
        
        if self.include_temporal_stats:
            # Temporal statistics
            x = np.arange(len(profile))
            if len(profile) > 1:
                slope, _ = np.polyfit(x, profile, 1)
                trend = np.corrcoef(x, profile)[0, 1] if len(profile) > 2 else 0
                variance = np.var(profile)
            else:
                slope = 0
                trend = 0
                variance = 0
            stats.extend([trend, slope, variance])
        
        if self.include_distribution_stats:
            # Distribution statistics
            from scipy import stats as scipy_stats
            try:
                skewness = scipy_stats.skew(profile)
                kurtosis = scipy_stats.kurtosis(profile)
            except:
                skewness = 0
                kurtosis = 0
            stats.extend([
                skewness,
                kurtosis,
                np.percentile(profile, 25),
                np.percentile(profile, 75)
            ])
        
        if self.include_peak_stats:
            # Peak statistics
            from scipy.signal import find_peaks
            try:
                peaks, _ = find_peaks(profile, height=np.mean(profile))
                peak_count = len(peaks)
                if peak_count > 0:
                    peak_values = profile[peaks]
                    peak_mean = np.mean(peak_values)
                    peak_std = np.std(peak_values)
                else:
                    peak_mean = 0
                    peak_std = 0
            except:
                peak_count = 0
                peak_mean = 0
                peak_std = 0
            stats.extend([peak_count, peak_mean, peak_std])
        
        return np.array(stats)
        
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names_.copy()


