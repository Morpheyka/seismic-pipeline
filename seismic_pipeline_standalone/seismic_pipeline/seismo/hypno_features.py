"""Hypnogram feature extraction helpers (FFT, staging, quality)."""
from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from scipy.stats import kurtosis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EllipticEnvelope as EE
from sklearn.mixture import GaussianMixture as GM

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
