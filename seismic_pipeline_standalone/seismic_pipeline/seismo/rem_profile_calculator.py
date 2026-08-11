"""
REM profile calculator for cached hypnogram data.

This module provides functionality to calculate REM sleep profiles
from cached hypnogram data and corresponding EEG signals.
"""

import warnings

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path
import logging
from .hypnogram_cache_manager import HypnogramCacheManagerYt
from ..mod.sklearnbaseyt import TransformerMixinYt


class REMProfileCalculatorYt(TransformerMixinYt):
    """
    Calculates REM sleep profiles from cached hypnogram data for a window of dates.
    For each input row (dict with 'rat_id' and 'window_dates'), calculates and concatenates
    the REM profiles for all dates in the window.
    """
    def __init__(self, 
                 cache_manager: Optional[HypnogramCacheManagerYt] = None,
                 window_size_hours: int = 6,  # Size of the sliding window in hours
                 step_size_hours: int = 1,    # Step size for the sliding window in hours
                 rem_stage: int = 2,  # REM stage value in hypnogram
                 n_points_per_day: int | None = None,
                 overlap: float = 0.0,
                 epoch_length_sec: int = 5,
                 sampling_rate: int = 250,
                 profile_features: List[str] = None,
                 fail_on_missing_data: bool = True):  # New parameter: raise exception if any hypnogram is missing
        super().__init__()
        self.cache_manager = cache_manager
        self.window_size_hours = window_size_hours
        self.step_size_hours = step_size_hours
        self.rem_stage = rem_stage
        self.n_points_per_day = n_points_per_day
        self.overlap = overlap
        self.epoch_length_sec = epoch_length_sec
        self.sampling_rate = sampling_rate
        self.profile_features = profile_features or ['rem_percentage']
        self.fail_on_missing_data = fail_on_missing_data
        self.max_length_ = 0  # Initialize max_length_

    def _calculate_features_for_X(
        self,
        X: List[Dict],
        *,
        return_day_lengths: bool = False,
    ) -> Tuple[List[np.ndarray], List[int]] | Tuple[List[np.ndarray], List[int], List[List[int]]]:
        """Helper to calculate features for a given X."""
        all_features = []
        valid_indices = []
        all_day_lengths: List[List[int]] = []
        missing_data_samples = []
        
        for i, row in enumerate(X):
            rat_id = row['rat_id']
            window_dates = row['window_dates']
            window_features = []
            day_lengths: List[int] = []
            has_valid_data = False
            missing_dates = []
            
            n_pts = self.n_points_per_day
            for date in window_dates:
                rem_profile = self._calculate_rem_profiles_for_rat_date(rat_id, date)
                if rem_profile.size > 0:
                    window_features.append(rem_profile)
                    day_lengths.append(int(rem_profile.size))
                    has_valid_data = True
                else:
                    missing_dates.append(date)
                    # Fixed-N mode: keep day-aligned slots so missing dates do not
                    # shift later days; fill with NaN of length n_points_per_day.
                    if n_pts is not None and int(n_pts) > 0:
                        window_features.append(
                            np.full(int(n_pts), np.nan, dtype=float)
                        )
                        day_lengths.append(0)

            if has_valid_data or (
                n_pts is not None and int(n_pts) > 0 and len(window_features) > 0
            ):
                window_features_concat = np.concatenate(window_features)
                all_features.append(window_features_concat)
                valid_indices.append(i)
                all_day_lengths.append(day_lengths)
            
            # Track samples with missing data
            if missing_dates:
                missing_data_samples.append((rat_id, missing_dates))
        
        # If fail_on_missing_data is True and we have missing data, raise exception
        if self.fail_on_missing_data and missing_data_samples:
            missing_info = "; ".join([f"{rat_id}: {', '.join(dates)}" for rat_id, dates in missing_data_samples])
            raise ValueError(f"Missing hypnogram data for: {missing_info}. "
                           f"This parameter combination cannot be evaluated.")
        
        if return_day_lengths:
            return all_features, valid_indices, all_day_lengths
        return all_features, valid_indices

    def fit(self, X, y=None):
        """Fit the transformer by determining the max feature length."""
        all_features, _ = self._calculate_features_for_X(X)
        if all_features:
            self.max_length_ = max(len(features) for features in all_features)
        else:
            self.max_length_ = 0
        return self

    def transform(self, X, y=None):
        """Transform the data by calculating and padding REM profiles."""
        all_features, valid_indices = self._calculate_features_for_X(X)
        
        if y is not None:
            y_filtered = y[valid_indices] if hasattr(y, '__getitem__') else y
        else:
            y_filtered = None

        if not all_features:
            # Return empty array with correct number of features if known
            return np.array([]).reshape(0, self.max_length_), y_filtered if y is not None else np.array([])

        padded_features = []
        metadata = []
        
        for i, features in enumerate(all_features):
            pad_width = self.max_length_ - len(features)
            if pad_width < 0:
                # This can happen if transform is called on data that has longer sequences than fit
                # Truncate in this case
                padded = features[:self.max_length_]
            else:
                padded = np.pad(features, (0, pad_width), 'constant')
            padded_features.append(padded)
            
            original_idx = valid_indices[i]
            metadata.append({
                'original_event_date': X[original_idx].get('original_event_date', ''),
                'original_rat_id': X[original_idx].get('original_rat_id', ''),
                'rat_id': X[original_idx]['rat_id'],
                'window_dates': X[original_idx]['window_dates']
            })
            
        self._last_metadata = metadata
        
        if not padded_features:
            return np.array([]).reshape(0, self.max_length_), y_filtered if y is not None else np.array([])
            
        return np.vstack(padded_features), y_filtered
        
    def _calculate_rem_profiles_for_rat_date(self, rat_id: str, date: str) -> np.ndarray:
        """
        Calculate REM profiles for a specific rat and date using the exact algorithm from Stage4DAG.py.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier
        date : str
            Date in YYYY_MM_DD format
            
        Returns
        -------
        np.ndarray
            REM profile values (percentages) for this rat/date
        """
        try:
            # ---- Negative cache: skip known-missing data instantly ----
            if self.cache_manager.is_known_missing(rat_id, date):
                return np.array([])

            # First try to get cached hypnogram
            hypnogram = self.cache_manager.get_cached_hypnogram(rat_id, date)
            
            # If not cached, try to cache it from S3 source first
            if hypnogram is None:
                if self.cache_manager._check_s3_temp_bucket_exists(rat_id, date):
                    success = self.cache_manager.cache_hypnogram(rat_id, date, 's3')
                else:
                    success = False
                if success:
                    hypnogram = self.cache_manager.get_cached_hypnogram(rat_id, date)
                else:
                    # If S3 fails, fallback to local source
                    success = self.cache_manager.cache_hypnogram(rat_id, date, 'local')
                    if success:
                        hypnogram = self.cache_manager.get_cached_hypnogram(rat_id, date)
                    
                    if hypnogram is None:
                        # Mark as missing so future lookups skip S3 entirely
                        self.cache_manager.mark_missing(rat_id, date)
                        return np.array([])
            
            # Validate hypnogram before processing
            if hypnogram is None:
                self.logger.warning(f"Hypnogram is None for {rat_id} on {date}")
                return np.array([])
            
            # Extract the actual hypnogram data (first element of the list)
            if isinstance(hypnogram, list):
                if len(hypnogram) > 0:
                    hypno_data = hypnogram[0]
                else:
                    self.logger.warning(f"Hypnogram list is empty for {rat_id} on {date}")
                    return np.array([])
            else:
                hypno_data = hypnogram
            
            # Validate hypno_data before processing
            if hypno_data is None:
                self.logger.warning(f"Hypnogram data is None for {rat_id} on {date}")
                return np.array([])
            
            # Ensure hypno_data is a numpy array
            if not isinstance(hypno_data, np.ndarray):
                try:
                    hypno_data = np.array(hypno_data)
                except Exception as e:
                    self.logger.error(f"Failed to convert hypnogram to array for {rat_id} on {date}: {e}")
                    return np.array([])
            
            # Validate array is not empty
            if len(hypno_data) == 0:
                self.logger.warning(f"Hypnogram array is empty for {rat_id} on {date}")
                return np.array([])
                
            if self.n_points_per_day is not None:
                return self.compute_rem_profile_fixed_n(
                    hypno_data,
                    self.n_points_per_day,
                    self.overlap,
                    self.rem_stage,
                )

            # Legacy hour-based sliding window (deprecated)
            rem_profile = self._fraction(
                hypno_data,
                (self.window_size_hours, self.step_size_hours),
                self.rem_stage,
                self.epoch_length_sec,
            )
            return np.array(rem_profile)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate REM profiles for {rat_id} on {date}: {e}")
            return np.array([])
            
    def compute_rem_profile_fixed_n(
        self,
        hypnogram: np.ndarray,
        n_points: int,
        overlap: float,
        rem_stage: int = 2,
    ) -> np.ndarray:
        """
        Compute REM profile with fixed number of points per day.

        Parameters
        ----------
        hypnogram : np.ndarray of shape (M,)
            Array of sleep stage labels (0=wake, 1=NREM, 2=REM) per 5-sec epoch.
        n_points : int
            Number of profile points per day (e.g., 12, 24, 48).
        overlap : float
            Overlap fraction in [0.0, 1.0). Window width = segment_length / (1 - overlap).
        rem_stage : int
            Label value for REM sleep (default 2).

        Returns
        -------
        profile : np.ndarray of shape (n_points,)
            REM fraction (0-100) at each point.
        """
        t = len(hypnogram)
        segment_len = t / n_points
        window_len = segment_len / (1.0 - overlap)

        profile = np.zeros(n_points)

        for i in range(n_points):
            center = (i + 0.5) * segment_len
            start = int(max(0, center - window_len / 2))
            end = int(min(t, center + window_len / 2))

            if end > start:
                window_epochs = hypnogram[start:end]
                rem_fraction = 100.0 * np.sum(window_epochs == rem_stage) / len(window_epochs)
            else:
                rem_fraction = 0.0

            profile[i] = rem_fraction

        return profile

    def _fraction(self, hypno: np.ndarray, slide_hours: Tuple[int, int], stage: int, window_sec: int) -> List[float]:
        """
        Calculate REM percentage using sliding window approach (exact algorithm from Stage4DAG.py).

        .. deprecated::
            Use :meth:`compute_rem_profile_fixed_n` with ``n_points_per_day`` and ``overlap``.

        Parameters
        ----------
        hypno : np.ndarray
            Hypnogram data
        slide_hours : tuple of int
            (window_size, step_size) in hours
        stage : int
            Sleep stage to calculate percentage for
        window_sec : int
            Length of each epoch in seconds

        Returns
        -------
        list of float
            REM percentages for each window
        """
        warnings.warn(
            "_fraction is deprecated; use compute_rem_profile_fixed_n with "
            "n_points_per_day and overlap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        points_hour = int(3600 / window_sec)
        res = []
        i = 0
        start = 0
        end = points_hour * slide_hours[0]
        
        while end < len(hypno):
            # Calculate percentage of specified stage in current window
            stage_count = len(np.where(hypno[start:end] == stage)[0])
            percentage = 100 * stage_count / (end - start)
            res.append(percentage)
            
            # Move window
            start += points_hour * slide_hours[1]
            end += points_hour * slide_hours[1]
            
        return res
            


class REMProfileCombinerYt(TransformerMixinYt):
    """
    Combines REM profiles from multiple days.
    
    This transformer takes REM profiles from multiple days and combines them
    into a unified representation for machine learning.
    """
    
    def __init__(self, 
                 combination_method: str = 'concatenate',
                 max_profiles_per_day: Optional[int] = None,
                 profile_features: List[str] = None):
        """
        Initialize the REM profile combiner.
        
        Parameters
        ----------
        combination_method : str, default='concatenate'
            Method for combining profiles ('concatenate', 'average', 'max')
        max_profiles_per_day : int, optional
            Maximum number of profiles to use per day
        profile_features : list of str, optional
            Features used in profiles
        """
        super().__init__()
        self.combination_method = combination_method
        self.max_profiles_per_day = max_profiles_per_day
        self.profile_features = profile_features or ['rem_percentage']
        
    def fit(self, X, y=None):
        """Fit the REM profile combiner."""
        return self
        
    def transform(self, X, y=None):
        """
        Combine REM profiles from multiple days.
        
        Parameters
        ----------
        X : array-like
            List of REM profiles for each day
        y : array-like, optional
            Labels
            
        Returns
        -------
        X_transformed : array-like
            Combined REM profiles
        y_transformed : array-like
            Labels (preserved)
        """
        combined_profiles = []
        
        for day_profile in X:
            if len(day_profile) == 0:
                # No REM profile for this day
                combined_profiles.append(np.array([0.0]))
                continue
                
            # day_profile is already a numpy array of REM percentages
            if self.combination_method == 'concatenate':
                combined = day_profile
            elif self.combination_method == 'average':
                combined = np.array([np.mean(day_profile)])
            elif self.combination_method == 'max':
                combined = np.array([np.max(day_profile)])
            elif self.combination_method == 'min':
                combined = np.array([np.min(day_profile)])
            elif self.combination_method == 'std':
                combined = np.array([np.std(day_profile)])
            elif self.combination_method == 'sum':
                combined = np.array([np.sum(day_profile)])
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")
                
            combined_profiles.append(combined)
            
        return np.array(combined_profiles), y
