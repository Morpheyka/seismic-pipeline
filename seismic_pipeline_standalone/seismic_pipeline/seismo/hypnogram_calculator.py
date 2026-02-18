"""
Hypnogram calculator transformer.

This module provides functionality to calculate hypnograms from EEG data
by downloading .dat files, processing them through a 3-stage pipeline,
and caching the resulting hypnograms.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import io
import os
import sys
from contextlib import contextmanager

from scipy.stats import kurtosis
from sklearn.covariance import EllipticEnvelope as EE
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.mixture import GaussianMixture as GM
from joblib import load as joblib_load

from .hypnogram_cache_manager import HypnogramCacheManagerYt
from .dat_file_cache_manager import DatFileCacheManagerYt
from .logging_config import get_mod_logger
from ..mod.sklearnbaseyt import TransformerMixinYt


# ============================================================================
# Helper functions adapted from local_pipeline.py
# ============================================================================


@contextmanager
def _sys_path(paths: List[str]):
    """
    Temporarily prepend paths to sys.path.

    This is needed to unpickle sklearn pipelines that reference custom modules
    from external research folders.
    """
    normalized = [str(Path(p).resolve()) for p in (paths or []) if p]
    old_path = list(sys.path)
    try:
        for path in reversed(normalized):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = old_path


class MLPChannelQualityPredictor:
    """
    Lazy wrapper for a persisted sklearn pipeline used for signal quality.

    The model is expected to accept rows in format:
    [date (YYYY_MM_DD), channel (1-based), rat_id].
    """

    def __init__(
        self,
        model_path: str,
        module_paths: Optional[List[str]] = None,
        good_classes: Tuple[int, ...] = (4, 5),
    ):
        self.model_path = str(model_path)
        self.module_paths = module_paths or []
        self.good_classes = tuple(int(v) for v in good_classes)
        self._model = None
        self._load_error: Optional[Exception] = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise self._load_error
        try:
            with _sys_path(self.module_paths):
                self._model = joblib_load(self.model_path)
            return self._model
        except Exception as e:
            self._load_error = e
            raise

    def predict_class(self, rat_id: str, date: str, channel_1b: int) -> int:
        model = self._load()
        pred = model.predict([[date, int(channel_1b), str(rat_id)]])[0]
        return int(pred)

    def is_good(self, rat_id: str, date: str, channel_1b: int) -> bool:
        pred_class = self.predict_class(rat_id, date, channel_1b)
        return pred_class in self.good_classes

def arr_to_epochs(arr, epoch_len_sec=5):
    """Split continuous data into epochs of epoch_len_sec seconds assuming 250 Hz."""
    epoch_points = int(250 * epoch_len_sec)
    eps = np.vstack(
        [arr[i * epoch_points : (i + 1) * epoch_points][np.newaxis] for i in range(int(len(arr) / epoch_points))]
    )
    return eps


def prepare_data(arr, channels, epoch_len_sec=5):
    """Ensure 2D (time x channels), select channels, convert to epochs."""
    arr = np.squeeze(arr)
    assert arr.ndim == 2, 'Wrong number of dimensions, expecting 2D'
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    arr = arr[:, channels]
    return arr_to_epochs(arr, epoch_len_sec)


def art_thr(art_rms, return_stat=False):
    """Estimate artifact threshold with Elliptic Envelope over RMS distribution."""
    n, b = np.histogram(art_rms, 200)
    b = (b[1:] + b[:-1]) / 2
    center = b[np.argmax(n[:150])]
    cont = 0.01
    thr = b[-1]
    kurts, thrs = [], []
    while (center < thr) and (cont < 0.5):
        y = EE(contamination=cont).fit_predict(art_rms.reshape((-1, 1)))
        kurts.append(kurtosis(art_rms[y == 1]))
        thrs.append((np.min(art_rms[y == -1]) + np.max(art_rms[y == 1])) / 2)
        thr = thrs[-1]
        cont += 0.01
    idx = np.argmin(np.abs(kurts))
    if return_stat:
        return thrs[idx], center, np.var(art_rms[art_rms < thrs[idx]]), np.percentile(art_rms, 99.5)
    else:
        return thrs[idx]


def art_thr_mad(x, k=7.0):
    """Robust MAD-based threshold: median + k * 1.4826 * MAD."""
    x = np.asarray(x).flatten()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return float(med + k * sigma)


def art_thr_quantile(x, q=99.5):
    """Quantile-based threshold (upper percentile)."""
    x = np.asarray(x).flatten()
    return float(np.percentile(x, q))


def art_thr_gmm(x):
    """Two-component GMM; threshold at probability crossover (≈ mode intersection)."""
    x = np.asarray(x).reshape(-1, 1)
    gmm = GM(n_components=2, covariance_type='full', reg_covar=1e-6).fit(x)
    xs = np.linspace(np.min(x), np.max(x), 2048).reshape(-1, 1)
    prob = gmm.predict_proba(xs)
    ix = np.argmin(np.abs(prob[:, 0] - prob[:, 1]))
    return float(xs[ix])


def art_thr_otsu(x, bins=256):
    """Otsu threshold on histogram; returns bin center that maximizes between-class variance."""
    x = np.asarray(x).flatten()
    hist, bin_edges = np.histogram(x, bins=bins)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_mids) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * bin_mids)[::-1] / np.maximum(weight2[::-1], 1e-12)))[::-1]

    inter_class_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if np.all(~np.isfinite(inter_class_var)) or inter_class_var.size == 0:
        return float(np.median(x))
    idx = np.argmax(inter_class_var)
    return float(bin_mids[idx])


class FFT_feature(TransformerMixin, BaseEstimator):
    def __init__(self, band=(0, 4), sf=250):
        self.sf = sf
        self.band = band

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        res = []
        for epoch in X:
            epoch = epoch.reshape((1, -1))
            fft = np.fft.fft(epoch, axis=-1)
            freq = np.fft.fftfreq(epoch.shape[-1], 1 / self.sf)
            spec_abs = np.abs(fft[:, freq >= 0]).reshape((fft.shape[0], -1))
            freq = freq[freq >= 0]
            res.append(np.sum(spec_abs[:, (freq >= self.band[0]) & (freq < self.band[1])], axis=-1).reshape((-1, 1)))
        return np.vstack(res) / 1e6


def delta_theta(eps, func, dband=(2.5, 5.5), tband=(5.5, 8)):
    tr = func(band=dband, sf=250)
    deltas = np.hstack([tr.transform(eps[:, :, ch]) for ch in range(eps.shape[2])])
    tr = func(band=tband, sf=250)
    ratios = np.hstack([tr.transform(eps[:, :, ch]) for ch in range(eps.shape[2])]) / deltas
    return deltas, ratios


def gm_delta(delta_day, cycles=2):
    points_cycle = int(delta_day.size / 2)
    res = []
    for c in range(cycles):
        delta = delta_day[c * points_cycle : (c + 1) * points_cycle].reshape((-1, 1))
        if len(delta) < 2:
            med = float(np.median(delta)) if len(delta) > 0 else 0.0
            mn = float(np.min(delta)) if len(delta) > 0 else 0.0
            mx = float(np.max(delta)) if len(delta) > 0 else 0.0
            res.append([med, mn, mx, 1e-3, 1e-3, mn, mx])
            continue
        try:
            gm_model = GM(n_components=2, covariance_type='full', reg_covar=1e-3).fit(delta)
            gm_preds = gm_model.predict(delta)
            shift = 0
            if gm_model.means_[0, 0] > gm_model.means_[1, 0]:
                shift = 1
                gm_preds = (gm_preds - 0.5) * (-1) + 0.5
            mask0 = (gm_preds == 0)
            mask1 = (gm_preds == 1)
            delta0 = delta[mask0]
            delta1 = delta[mask1]
            if delta0.size == 0 or delta1.size == 0:
                median_val = np.median(delta)
                delta0 = delta[delta.flatten() <= median_val]
                delta1 = delta[delta.flatten() > median_val]
                if delta0.size == 0 or delta1.size == 0:
                    mid = len(delta) // 2
                    delta0 = delta[:mid]
                    delta1 = delta[mid:]
            max0 = np.max(delta0) if delta0.size > 0 else np.min(delta)
            min1 = np.min(delta1) if delta1.size > 0 else np.max(delta)
            mean0 = np.mean(delta0) if delta0.size > 0 else np.min(delta)
            mean1 = np.mean(delta1) if delta1.size > 0 else np.max(delta)
            res.append([
                (max0 + min1) / 2,
                mean0,
                mean1,
                gm_model.covariances_.flatten()[0 - shift],
                gm_model.covariances_.flatten()[1 - shift],
                np.min(delta),
                np.max(delta),
            ])
        except Exception:
            med = float(np.median(delta))
            mn = float(np.min(delta))
            mx = float(np.max(delta))
            res.append([med, mn, mx, 1e-3, 1e-3, mn, mx])
    return np.array(res).reshape((cycles, 7))


def dc_remove(data, window_sec=50, fs=0.2):
    i, res, baseline = 0, [], []
    window = round(window_sec * fs)
    while i < len(data):
        baseline.append(np.min(data[max(0, i - window) : min(len(data) - 1, i + window)]))
        res.append(data[i] - baseline[-1])
        i += 1
    return np.array(res).flatten()


def no_singles(ser, val):
    res = [ser[0]]
    for i in range(1, len(ser) - 1):
        if (ser[i] == val) and (ser[i - 1] == ser[i + 1]) and (ser[i - 1] != val):
            res.append(ser[i - 1])
        else:
            res.append(ser[i])
    res.append(ser[-1])
    return np.array(res)


def no_rems_in_wake(hypno, n_back):
    res = [*hypno[:n_back]]
    i = n_back
    while i < len(hypno):
        if (hypno[i] == 2) and (np.sum(hypno[i - n_back : i]) == 0):
            res.append(0)
            j = i + 1
            while j < len(hypno) and hypno[j] == 2:
                res.append(0)
                j += 1
                if j == len(hypno):
                    break
            i = j
        else:
            res.append(hypno[i])
            i += 1
    return np.array(res)


def no_single_a_between_b_and_c(hypno, a, b, c):
    res = [hypno[0]]
    for i in range(1, len(hypno) - 1):
        if (hypno[i] == a) and (hypno[i - 1] == b) and (hypno[i + 1] == c):
            res.append(b)
        else:
            res.append(hypno[i])
    res.append(hypno[-1])
    return np.array(res)


def stage(art_masks, deltas, ratios, params1, params2):
    hypnos = np.zeros(len(art_masks))
    norm_mask = np.logical_not(art_masks)
    n_cycles = max([len(v) for v in params1.values()] + [len(v) for v in params2.values()]) if params1 else 1
    n_pts_cycle = int(len(art_masks) / n_cycles) if n_cycles > 0 else len(art_masks)
    for c in range(n_cycles):
        cycle_slice = slice(c * n_pts_cycle, (c + 1) * n_pts_cycle)
        art_mask = norm_mask[cycle_slice]
        part_hypno = hypnos[cycle_slice][art_mask]
        mask1 = np.zeros(len(part_hypno))
        nan_ch = 0
        for ch in params1:
            if np.isnan(params1[ch][min(c, len(params1[ch]) - 1)]):
                nan_ch += 1
            else:
                mask1 += (deltas[cycle_slice, ch][art_mask] > params1[ch][min(c, len(params1[ch]) - 1)]).astype(int)
        if len(params1) - nan_ch > 0:
            part_hypno = (mask1 / (len(params1) - nan_ch)).round()
        else:
            part_hypno = mask1
        mask2 = np.zeros(len(mask1))
        for ch in params2:
            mask2 += (ratios[cycle_slice, ch][art_mask] > params2[ch][min(c, len(params2[ch]) - 1)]).astype(int)
        if len(params2) > 0:
            mask2 = (mask2 / len(params2)).round()
        part_hypno[(mask2 == 1)] = 2
        hypnos[cycle_slice][art_mask] = part_hypno

    hypno = hypnos
    dummy = hypno.copy()
    j = 0
    not_mask = np.argwhere(art_masks).flatten()
    while j < len(not_mask):
        idx = not_mask[j]
        k = idx
        while (k - idx) < 1:
            if idx > 0:
                dummy[k] = dummy[idx - 1]
            k += 1
            if k >= len(not_mask):
                break
        j = k
    hypno = dummy
    hypno = no_singles(hypno, 0)
    hypno = no_singles(hypno, 1)
    hypno = no_rems_in_wake(hypno, 2)
    hypno = no_single_a_between_b_and_c(hypno, 0, 1, 2)
    hypno = no_single_a_between_b_and_c(hypno, 1, 2, 2)

    return hypno


# ============================================================================
# Main Transformer Class
# ============================================================================

class HypnoCalculatorYt(TransformerMixinYt):
    """
    Calculates hypnograms from EEG data for a window of dates.
    
    For each input row (dict with 'rat_id' and 'window_dates'), downloads .dat files,
    processes them through a 3-stage pipeline, and caches the resulting hypnograms.
    Returns the exact same X and y as input (pass-through transformer).
    """
    
    def __init__(self, 
                 cache_manager: Optional[HypnogramCacheManagerYt] = None,
                 dat_cache_manager: Optional[DatFileCacheManagerYt] = None,
                 use_s3_dat: bool = False,
                 local_data_root: str = '/mnt/wd/rat',
                 s3_config: Optional[Dict] = None,
                 epoch_length_sec: int = 5,
                 metric: str = 'square',  # 'square' or 'abs'
                 threshold: str = 'GMM',  # 'GMM' or 'MAD'
                 channel_quality_model: Optional[Any] = None,
                 quality_model_path: Optional[str] = None,
                 quality_model_module_paths: Optional[List[str]] = None,
                 quality_good_classes: Tuple[int, ...] = (4, 5),
                 quality_fallback_to_all_channels: bool = False,
                 channel_cutoff_date: str = '2025_06_01'):
        """
        Initialize the hypnogram calculator.
        
        Parameters
        ----------
        cache_manager : HypnogramCacheManagerYt, optional
            Cache manager for hypnograms
        dat_cache_manager : DatFileCacheManagerYt, optional
            Cache manager for .dat files
        use_s3_dat : bool, default=False
            Whether to load .dat files from S3
        local_data_root : str, default='/mnt/wd/rat'
            Local path for .dat files
        s3_config : dict, optional
            S3 connection configuration
        epoch_length_sec : int, default=5
            Epoch length in seconds
        metric : str, default='square'
            Artifact metric: 'square' (eps**2) or 'abs' (abs(eps))
        threshold : str, default='GMM'
            Artifact threshold method: 'GMM' or 'MAD'
        channel_quality_model : Any, optional
            Loaded model object or predictor-like object for channel quality prediction
        quality_model_path : str, optional
            Path to persisted quality model pickle/joblib.
        quality_model_module_paths : list of str, optional
            Additional module search paths required by model unpickling.
        quality_good_classes : tuple of int, default=(4, 5)
            Predicted classes considered high quality.
        quality_fallback_to_all_channels : bool, default=False
            If True and quality inference fails, do not filter channels by quality.
        channel_cutoff_date : str, default='2025_06_01'
            Date threshold for channel selection (after: 2 channels, before: 4 channels)
        """
        super().__init__()
        self.cache_manager = cache_manager
        self.dat_cache_manager = dat_cache_manager
        self.use_s3_dat = use_s3_dat
        self.local_data_root = local_data_root
        self.s3_config = s3_config
        self.epoch_length_sec = epoch_length_sec
        self.metric = metric
        self.threshold = threshold
        self.channel_quality_model = channel_quality_model
        self.quality_model_path = quality_model_path
        self.quality_model_module_paths = quality_model_module_paths or []
        self.quality_good_classes = tuple(int(v) for v in quality_good_classes)
        self.quality_fallback_to_all_channels = quality_fallback_to_all_channels
        self.channel_cutoff_date = channel_cutoff_date
        self._quality_predictor: Optional[MLPChannelQualityPredictor] = None
        
        # Initialize dat cache manager if not provided
        if self.dat_cache_manager is None:
            self.dat_cache_manager = DatFileCacheManagerYt(
                local_data_root=local_data_root,
                s3_config=s3_config
            )
        
        self.logger = get_mod_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Validate parameters
        if metric not in ['square', 'abs']:
            raise ValueError(f"Invalid metric '{metric}'. Must be 'square' or 'abs'")
        if threshold not in ['GMM', 'MAD']:
            raise ValueError(f"Invalid threshold '{threshold}'. Must be 'GMM' or 'MAD'")
        if len(self.quality_good_classes) == 0:
            raise ValueError("quality_good_classes must contain at least one class label")
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op, stateless)."""
        return self
    
    def transform(self, X, y=None):
        """
        Transform the data by calculating and caching hypnograms.
        
        Returns the exact same X and y as input (pass-through).
        """
        if not X:
            return X, y
        
        # Process each unique rat/date combination
        processed_dates = set()
        
        for row in X:
            rat_id = row['rat_id']
            window_dates = row['window_dates']
            
            for date in window_dates:
                # Skip if already processed
                cache_key = f"{rat_id}_{date}"
                if cache_key in processed_dates:
                    continue
                
                # Check if hypnogram already cached
                if self.cache_manager:
                    cached_hypno = self.cache_manager.get_cached_hypnogram(rat_id, date)
                    if cached_hypno is not None:
                        self.logger.debug(f"Hypnogram already cached for {rat_id} on {date}")
                        processed_dates.add(cache_key)
                        continue
                
                # Calculate hypnogram
                try:
                    # Pre-cache/load DAT before computation path (local -> S3 fallback).
                    source = 's3' if self.use_s3_dat else 'local'
                    eeg_data = self.dat_cache_manager.get_cached_dat_file(rat_id, date, source=source)

                    if eeg_data is None or eeg_data.size == 0:
                        self.logger.warning(
                            f"No EEG DAT available for {rat_id} on {date}; "
                            "trying existing hypnogram fallback from local/S3"
                        )
                        created = self._fallback_to_existing_hypnogram(rat_id, date)
                    else:
                        created = self._calculate_hypnogram(rat_id, date, eeg_data=eeg_data)

                    # Mark as processed even if explicitly skipped to avoid retries
                    processed_dates.add(cache_key)
                    if not created:
                        self.logger.info(
                            f"Could not compute or fallback hypnogram for {rat_id} on {date}"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to calculate hypnogram for {rat_id} on {date}: {e}")
                    continue
        
        # Return exact same X and y
        return X, y
    
    def _calculate_hypnogram(self, rat_id: str, date: str, eeg_data: Optional[np.ndarray] = None) -> bool:
        """Calculate hypnogram for a specific rat and date.

        Returns True when a hypnogram was created and cached, False when
        creation is not possible and fallback also fails.
        """
        # Load .dat file if caller didn't pre-cache it.
        if eeg_data is None:
            source = 's3' if self.use_s3_dat else 'local'
            eeg_data = self.dat_cache_manager.get_cached_dat_file(rat_id, date, source=source)

        if eeg_data is None or eeg_data.size == 0:
            self.logger.warning(
                f"No EEG data available for {rat_id} on {date}; "
                "trying existing hypnogram fallback from local/S3"
            )
            return self._fallback_to_existing_hypnogram(rat_id, date)
        
        # Determine channels based on date
        channels = self._get_channels_for_date(date)
        
        # Keep only high-quality channels (predicted classes in self.quality_good_classes)
        channel_quality = self._get_channel_quality(rat_id, date, eeg_data, channels)
        selected_channels = [ch for ch in channels if channel_quality.get(ch, -1) in self.quality_good_classes]
        if len(selected_channels) == 0:
            self.logger.warning(
                f"No high-quality channels for {rat_id} on {date}; "
                "trying existing hypnogram fallback from local/S3"
            )
            return self._fallback_to_existing_hypnogram(rat_id, date)
        
        # Run 3-stage pipeline (in-memory CSVs)
        art_thrs_df = self._prepare_thr_pics(eeg_data, rat_id, selected_channels, channel_quality)
        
        if art_thrs_df is None or art_thrs_df.empty or np.all(art_thrs_df['Accept'].values == 0):
            self.logger.warning(
                f"All selected channels rejected by artifact stage for {rat_id} on {date}; "
                "trying existing hypnogram fallback from local/S3"
            )
            return self._fallback_to_existing_hypnogram(rat_id, date)
        else:
            delta_thrs_df, ratio_thrs_df = self._prepare_theta_delta(
                eeg_data, rat_id, selected_channels, art_thrs_df, channel_quality
            )
            hypnogram = self._score(
                eeg_data, rat_id, selected_channels, art_thrs_df, delta_thrs_df, ratio_thrs_df, channel_quality
            )
        
        # Cache hypnogram directly to hypnogram cache (works even when local_data_root is read-only)
        if self.cache_manager:
            success = self.cache_manager.cache_hypnogram_from_data(rat_id, date, hypnogram)
            if not success:
                self.logger.warning(f"Failed to cache hypnogram for {rat_id} on {date}")
                return False
            # Optionally write to local_data_root for persistence (best effort)
            try:
                date_parsed = self._parse_date(date)
                expected_path = Path(self.local_data_root) / date_parsed / f"{rat_id}_hypno.pickle"
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                import pickle
                with open(expected_path, 'wb') as f:
                    pickle.dump(hypnogram, f)
            except (OSError, PermissionError) as e:
                self.logger.debug(f"Could not write hypnogram to {expected_path}: {e} (cache is populated)")
        
        self.logger.info(f"Calculated and cached hypnogram for {rat_id} on {date}")
        return True

    def _fallback_to_existing_hypnogram(self, rat_id: str, date: str) -> bool:
        """
        Try to resolve missing/insufficient compute case by caching existing hypnogram.

        Source order:
        1) local WD path
        2) S3 temp bucket
        """
        if self.cache_manager is None:
            self.logger.warning(
                f"No cache manager configured for fallback hypnogram lookup ({rat_id} {date})"
            )
            return False

        if self.cache_manager.cache_hypnogram(rat_id, date, source='local'):
            self.logger.info(f"Fallback hypnogram resolved from local for {rat_id} on {date}")
            return True

        if self.cache_manager.cache_hypnogram(rat_id, date, source='s3'):
            self.logger.info(f"Fallback hypnogram resolved from S3 for {rat_id} on {date}")
            return True

        self.logger.warning(f"No fallback hypnogram found in local/S3 for {rat_id} on {date}")
        return False
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to YYYY_MM_DD format."""
        if '_' in date_str:
            return date_str
        elif '-' in date_str:
            return date_str.replace('-', '_')
        else:
            return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"
    
    def _get_channels_for_date(self, date: str) -> List[int]:
        """
        Get channel list based on date.
        
        If date >= channel_cutoff_date: use 2 channels [0, 1]
        If date < channel_cutoff_date: use 4 channels [0, 1, 2, 3]
        """
        date_parsed = self._parse_date(date)
        cutoff_parsed = self._parse_date(self.channel_cutoff_date)
        
        # Compare dates
        if date_parsed >= cutoff_parsed:
            return [0, 1]  # 2 channels
        else:
            return [0, 1, 2, 3]  # 4 channels
    
    def _get_channel_quality(
        self,
        rat_id: str,
        date: str,
        eeg_data: np.ndarray,
        channels: List[int],
    ) -> Dict[int, int]:
        """
        Predict quality class for each requested channel.
        
        Returns:
            dict mapping channel index (0-based) to predicted class (int)
        """
        if not channels:
            return {}

        date_parsed = self._parse_date(date)

        # No model configured -> preserve legacy behavior (accept all requested channels)
        if self.channel_quality_model is None and not self.quality_model_path:
            default_good = max(self.quality_good_classes)
            return {ch: default_good for ch in channels}

        quality: Dict[int, int] = {}
        try:
            for ch in channels:
                ch_1b = int(ch) + 1
                # 1) Explicit predictor object
                if self.channel_quality_model is not None:
                    predictor = self.channel_quality_model
                    if hasattr(predictor, "predict_class"):
                        pred_class = int(predictor.predict_class(rat_id, date_parsed, ch_1b))
                    elif hasattr(predictor, "is_good"):
                        pred_class = max(self.quality_good_classes) if predictor.is_good(rat_id, date_parsed, ch_1b) else 0
                    elif hasattr(predictor, "predict"):
                        pred_class = int(predictor.predict([[date_parsed, ch_1b, str(rat_id)]])[0])
                    else:
                        raise TypeError("channel_quality_model does not expose predict/predict_class/is_good")
                else:
                    # 2) Lazy-loaded model from pickle/joblib
                    if self._quality_predictor is None:
                        self._quality_predictor = MLPChannelQualityPredictor(
                            model_path=self.quality_model_path,
                            module_paths=self.quality_model_module_paths,
                            good_classes=self.quality_good_classes,
                        )
                    pred_class = self._quality_predictor.predict_class(rat_id, date_parsed, ch_1b)

                quality[ch] = int(pred_class)

        except Exception as e:
            self.logger.warning(f"Channel quality inference failed for {rat_id} on {date}: {e}")
            if self.quality_fallback_to_all_channels:
                default_good = max(self.quality_good_classes)
                return {ch: default_good for ch in channels}
            return {ch: 0 for ch in channels}

        return quality
    
    def _prepare_thr_pics(self, arr: np.ndarray, rat: str, channels: List[int], 
                         channel_quality: Dict[int, float]) -> Optional[pd.DataFrame]:
        """Stage 1: Compute artifact thresholds (in-memory)."""
        eps = prepare_data(arr, channels, self.epoch_length_sec)
        
        # Calculate artifacts based on metric
        if self.metric == 'abs':
            arts = np.max(np.abs(eps), axis=1)
        else:  # square
            arts = np.max(eps**2, axis=1)
        
        art_thrs = []
        
        for ch in range(arts.shape[1]):
            series = arts[:, ch]
            
            # Compute thresholds
            thr_ee = art_thr(series)
            thr_mad = art_thr_mad(series)
            thr_qtl = art_thr_quantile(series)
            try:
                thr_gmm = art_thr_gmm(series)
            except Exception:
                thr_gmm = np.nan
            thr_otsu = art_thr_otsu(series)
            
            # Choose threshold based on self.threshold
            if self.threshold == 'GMM':
                thr = thr_gmm if np.isfinite(thr_gmm) else thr_otsu
            else:  # MAD
                thr = thr_mad
            
            art_thrs.append(thr)
        
        # Create DataFrame in memory (instead of writing to CSV)
        art_thrs_df = pd.DataFrame(
            data=np.stack(([1] * len(art_thrs), art_thrs), axis=1),
            columns=('Accept', 'Threshold')
        )
        
        return art_thrs_df
    
    def _prepare_theta_delta(self, arr: np.ndarray, rat: str, channels: List[int],
                            art_thrs_df: pd.DataFrame, channel_quality: Dict[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 2: Compute delta/theta features and thresholds (in-memory)."""
        if art_thrs_df.empty or np.all(art_thrs_df['Accept'].values == 0):
            # Return empty DataFrames
            return pd.DataFrame(columns=('Accept', 'Threshold')), pd.DataFrame(columns=('Accept', 'Threshold'))
        
        eps = prepare_data(arr, channels, self.epoch_length_sec)
        
        # Calculate artifacts
        if self.metric == 'abs':
            arts = np.squeeze(np.max(np.abs(eps), axis=1))
        else:  # square
            arts = np.squeeze(np.max(eps**2, axis=1))
        
        arts_params = {idx: art_thrs_df.iloc[idx, 1] for idx in np.argwhere(art_thrs_df['Accept'].values == 1).flatten()}
        
        # Combine artifact masks
        _masks = [arts[:, ch] > thr for ch, thr in arts_params.items()]
        if len(_masks) == 0:
            art_mask = np.zeros(arts.shape[0], dtype=bool)
        elif len(_masks) == 1:
            art_mask = _masks[0]
        else:
            art_mask = np.logical_or.reduce(_masks)
        
        deltas, ratios = delta_theta(eps, FFT_feature)
        
        cycles = 2
        delta_thrs_gmm = []
        
        for ch in range(deltas.shape[1]):
            deltas[:, ch] = np.convolve(np.convolve(deltas[:, ch].flatten(), np.ones(5) / 5, mode='same'), np.ones(5) / 5, mode='same')
            clean_deltas = deltas[:, ch][np.logical_not(art_mask)]
            if clean_deltas.size < 2:
                clean_deltas = deltas[:, ch]
            
            try:
                p99 = np.percentile(clean_deltas, 99)
                clean_for_thr = clean_deltas[clean_deltas <= p99]
                if clean_for_thr.size < 2:
                    clean_for_thr = clean_deltas
            except Exception:
                clean_for_thr = clean_deltas
            
            points_cycle = max(int(clean_for_thr.size / cycles), 1)
            for i in range(cycles):
                cyc = clean_for_thr[i * points_cycle : (i + 1) * points_cycle]
                if cyc.size == 0:
                    cyc = clean_for_thr
                try:
                    gmm_thr = gm_delta(cyc, cycles=1)[:, 0][0]
                except Exception:
                    gmm_thr = art_thr_otsu(cyc)
                delta_thrs_gmm.append(float(gmm_thr))
        
        # Create delta thresholds DataFrame
        delta_thrs_df = pd.DataFrame(
            data=np.stack(([1] * len(delta_thrs_gmm), delta_thrs_gmm), axis=1),
            columns=('Accept', 'Threshold')
        )
        
        # Ratio thresholds
        ratio_thrs = []
        for ch in range(ratios.shape[1]):
            ratios[:, ch] = np.convolve(ratios[:, ch].flatten(), np.ones(5) / 5, mode='same')
            ratios[:, ch] = dc_remove(ratios[:, ch], 100)
            clean_ratios = ratios[:, ch][np.logical_not(art_mask)]
            if clean_ratios.size < 2:
                clean_ratios = ratios[:, ch]
            ratio_thrs.append(art_thr(clean_ratios, True)[0])
        
        # Create ratio thresholds DataFrame
        ratio_thrs_df = pd.DataFrame(
            data=np.stack(([1] * len(ratio_thrs), ratio_thrs), axis=1),
            columns=('Accept', 'Threshold')
        )
        
        return delta_thrs_df, ratio_thrs_df
    
    def _score(self, arr: np.ndarray, rat: str, channels: List[int],
               art_thrs_df: pd.DataFrame, delta_thrs_df: pd.DataFrame, 
               ratio_thrs_df: pd.DataFrame, channel_quality: Dict[int, float]) -> np.ndarray:
        """Stage 3: Classify sleep stages, generate hypnogram."""
        if art_thrs_df.empty or np.all(art_thrs_df['Accept'].values == 0):
            return np.full(17280, -1)
        
        eps = prepare_data(arr, channels, self.epoch_length_sec)
        
        # Calculate artifacts
        if self.metric == 'abs':
            arts = np.max(np.abs(eps), axis=1)
        else:  # square
            arts = np.max(eps**2, axis=1)
        
        # Compute thresholds for both GMM and MAD methods
        arts_params_gmm = {}
        arts_params_mad = {}
        accepted_channels = np.argwhere(art_thrs_df['Accept'].values == 1).flatten()
        
        for ch_idx in accepted_channels:
            series = arts[:, ch_idx]
            # GMM threshold
            try:
                thr_gmm = art_thr_gmm(series)
                if np.isfinite(thr_gmm):
                    arts_params_gmm[ch_idx] = thr_gmm
            except Exception:
                pass
            # MAD threshold
            thr_mad = art_thr_mad(series)
            arts_params_mad[ch_idx] = thr_mad
        
        # Create art masks based on threshold method
        if self.threshold == 'GMM':
            arts_params = arts_params_gmm
        else:  # MAD
            arts_params = arts_params_mad
        
        # Use AND of first 2 channels if multiple, or single channel
        cum_and = lambda a: np.logical_and(*a[:2])
        if len(arts_params) > 1:
            art_mask = cum_and([arts[:, ch] > thr for ch, thr in arts_params.items()])
        elif len(arts_params) == 1:
            ch_key = list(arts_params.keys())[0]
            art_mask = arts[:, ch_key] > arts_params[ch_key]
        else:
            art_mask = np.zeros(arts.shape[0], dtype=bool)
        
        deltas, ratios = delta_theta(eps, FFT_feature)
        for ch in range(deltas.shape[1]):
            deltas[:, ch] = np.convolve(np.convolve(deltas[:, ch].flatten(), np.ones(5) / 5, mode='same'), np.ones(5) / 5, mode='same')
            ratios[:, ch] = np.convolve(ratios[:, ch].flatten(), np.ones(5) / 5, mode='same')
            ratios[:, ch] = dc_remove(ratios[:, ch], 100)
        
        params1, params2 = {}, {}
        cycles = int(len(delta_thrs_df) / deltas.shape[1]) if len(delta_thrs_df) > 0 else 1
        
        for ch in range(deltas.shape[1]):
            if ch * cycles < len(delta_thrs_df):
                ch_delta = delta_thrs_df.iloc[ch * cycles : ch * cycles + cycles]
                if np.all(ch_delta['Accept'].values == 1):
                    params1[ch] = ch_delta['Threshold'].values.tolist()
                elif np.any(ch_delta['Accept'].values == 1):
                    dumm = ch_delta['Threshold'].values.copy()
                    dumm[ch_delta['Accept'].values != 1] = np.nan
                    params1[ch] = dumm.tolist()
            
            if ch < len(ratio_thrs_df):
                ch_ratio = ratio_thrs_df.iloc[ch]
                if ch_ratio['Accept'] == 1:
                    params2[ch] = [ch_ratio['Threshold']]
        
        # Generate hypnogram
        hypno = stage(art_mask, deltas, ratios, params1, params2)
        
        return hypno

