"""Auto-split from rem_profiles_export_10days_lib.py."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

from seismic_pipeline.features.runtime import (
    get_runtime_data_raw,
    get_runtime_export_cfg,
    get_runtime_prepare_cfg,
    set_runtime_data_norm,
    set_runtime_data_raw,
    set_runtime_prepare_cfg,
)

_SHAPE_SHIFT_METRIC = "shape_shift"
_SHAPE_SHIFT_GROUPS = frozenset({"all", "concat", "odd", "even"})
FIXED_N_CHUNK_DAYS = 8

def maxmin_scale(row: np.ndarray) -> np.ndarray:
    """Normalize a 1D row to [0, 1] ignoring NaN padding."""
    finite = np.isfinite(row)
    res = np.full_like(row, np.nan, dtype=float)

    if not np.any(finite):
        return res

    rmin = float(np.min(row[finite]))
    rmax = float(np.max(row[finite]))

    if rmax == rmin:
        res[finite] = 0.5
        return res

    res[finite] = (row[finite] - rmin) / (rmax - rmin)
    return res
def load_and_normalize(csv_path: str = "samples_10days_nanpad.csv"):
    """Load CSV and normalize each row independently (NaN-aware)."""
    df = pd.read_csv(csv_path, header=None, na_values=["", "NaN", "nan"], keep_default_na=True)
    data_raw = df.to_numpy(dtype=float)
    data_norm = np.array([maxmin_scale(row) for row in data_raw], dtype=float)
    return data_raw, data_norm


def _guess_day_lengths_csv(csv_path: str) -> str | None:
    candidates = [
        csv_path.replace("_nanpad.csv", "_nanpad_day_lengths.csv"),
        csv_path.replace("nanpad.csv", "nanpad_day_lengths.csv"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _guess_metadata_csv(csv_path: str) -> str | None:
    candidates = [
        csv_path.replace("_nanpad.csv", "_metadata.csv"),
        csv_path.replace("nanpad.csv", "metadata.csv"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _parse_day_lengths_cell(cell) -> list[int]:
    text = "" if cell is None or (isinstance(cell, float) and np.isnan(cell)) else str(cell).strip()
    if not text:
        return []
    return [int(x) for x in text.split(";") if str(x).strip()]


def _exported_metadata_rows(csv_path: str, n_samples: int) -> pd.DataFrame:
    meta_path = _guess_metadata_csv(csv_path)
    if meta_path is None:
        raise FileNotFoundError(
            f"Metadata CSV not found for {csv_path!r}. "
            "Expected a sibling *_metadata.csv file."
        )
    meta = pd.read_csv(meta_path)
    if "exported" not in meta.columns:
        raise ValueError(f"Metadata file {meta_path!r} has no 'exported' column.")
    exported = meta[meta["exported"].astype(bool)].sort_values("row_index").reset_index(drop=True)
    if len(exported) != n_samples:
        raise ValueError(
            f"Metadata exported rows ({len(exported)}) != CSV rows ({n_samples}) "
            f"for {meta_path!r}"
        )
    return exported


def _write_day_lengths_sidecar(sidecar_path: str, day_lengths: list[list[int]]) -> None:
    pd.DataFrame(
        {
            "day_profile_lengths": [
                ";".join(str(int(x)) for x in lengths) for lengths in day_lengths
            ]
        }
    ).to_csv(sidecar_path, index=False)


def _load_day_lengths_from_metadata(csv_path: str, n_samples: int) -> list[list[int]] | None:
    meta_path = _guess_metadata_csv(csv_path)
    if meta_path is None or not os.path.isfile(meta_path):
        return None
    meta = pd.read_csv(meta_path)
    if "day_profile_lengths" not in meta.columns:
        return None
    exported = meta[meta["exported"].astype(bool)].sort_values("row_index").reset_index(drop=True)
    if len(exported) != n_samples:
        return None
    out: list[list[int]] = []
    for cell in exported["day_profile_lengths"].tolist():
        parsed = _parse_day_lengths_cell(cell)
        if not parsed:
            return None
        out.append(parsed)
    return out


def _compute_day_lengths_from_hypnogram_cache(
    csv_path: str,
    n_samples: int,
    export_cfg: dict | None,
) -> list[list[int]]:
    """Recompute per-day REM profile lengths from cached hypnograms + metadata."""
    from seismic_pipeline.config.changepoint_defaults import s3_config_from_env
    from seismic_pipeline.seismo.hypnogram_cache_manager import HypnogramCacheManagerYt
    from seismic_pipeline.seismo.rem_profile_calculator import REMProfileCalculatorYt

    exported = _exported_metadata_rows(csv_path, n_samples)
    cfg = dict(export_cfg or {})
    cache_manager = HypnogramCacheManagerYt(
        local_cache_dir=str(cfg.get("local_hypnogram_cache_dir", "./hypnogram_cache_legacy10")),
        local_data_root=str(cfg.get("local_data_root", "/mnt/wd/rat")),
        s3_config=cfg.get("s3_config") or s3_config_from_env(),
        s3_rat_bucket=str(cfg.get("s3_rat_bucket", "rat")),
        s3_temp_bucket=str(cfg.get("s3_temp_bucket", "temp")),
        allow_local_root_fallback=True,
    )
    rem_calc = REMProfileCalculatorYt(
        cache_manager=cache_manager,
        n_points_per_day=int(cfg.get("n_points_per_day")) if cfg.get("n_points_per_day") is not None else None,
        overlap=float(cfg.get("overlap", 0.0)),
        window_size_hours=int(cfg.get("window_size_hours", 6)),
        step_size_hours=int(cfg.get("step_size_hours", 1)),
        rem_stage=int(cfg.get("rem_stage", 2)),
        epoch_length_sec=int(cfg.get("epoch_length_sec", 5)),
        sampling_rate=int(cfg.get("sampling_rate", 250)),
        fail_on_missing_data=False,
    )

    out: list[list[int]] = []
    for _, row in exported.iterrows():
        rat_id = str(row["rat_id"])
        dates = [d.strip() for d in str(row["window_dates"]).split(";") if d.strip()]
        lengths: list[int] = []
        for date in dates:
            prof = rem_calc._calculate_rem_profiles_for_rat_date(rat_id, date)
            lengths.append(int(prof.size) if prof.size > 0 else 0)
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError(
                f"Missing daily REM profiles for rat={rat_id}, dates={dates!r}"
            )
        out.append(lengths)
    return out


def _resolve_day_lengths_per_sample(
    csv_path: str,
    n_samples: int,
    *,
    export_cfg: dict | None = None,
    write_sidecar: bool = True,
) -> list[list[int]] | None:
    """Load or compute per-row day profile lengths aligned with CSV rows."""
    lengths = _load_day_lengths_per_sample(csv_path, n_samples)
    if lengths is not None:
        return lengths

    lengths = _load_day_lengths_from_metadata(csv_path, n_samples)
    if lengths is not None:
        return lengths

    cfg = export_cfg if export_cfg is not None else get_runtime_export_cfg()
    try:
        lengths = _compute_day_lengths_from_hypnogram_cache(csv_path, n_samples, cfg)
    except Exception as exc:
        warnings.warn(
            f"Could not resolve day profile lengths for shape_shift ({exc}). "
            "Falling back to equal-width day splits, which may fail when daily "
            "profile lengths vary. Re-export REM profiles or ensure hypnogram cache "
            "and metadata CSV are available.",
            stacklevel=2,
        )
        return None

    if write_sidecar:
        sidecar = _guess_day_lengths_csv(csv_path)
        if sidecar is None:
            sidecar = csv_path.replace("_nanpad.csv", "_nanpad_day_lengths.csv")
        _write_day_lengths_sidecar(sidecar, lengths)
        print(f"Saved day profile lengths sidecar: {sidecar}")

    return lengths


def _load_day_lengths_per_sample(csv_path: str, n_samples: int) -> list[list[int]] | None:
    sidecar = _guess_day_lengths_csv(csv_path)
    if sidecar is None:
        return None
    df = pd.read_csv(sidecar)
    if "day_profile_lengths" not in df.columns:
        return None
    if len(df) != n_samples:
        raise ValueError(
            f"Day-length sidecar rows ({len(df)}) != CSV rows ({n_samples}) for {sidecar!r}"
        )
    out: list[list[int]] = []
    for cell in df["day_profile_lengths"].tolist():
        out.append(_parse_day_lengths_cell(cell))
    return out
def _split_row_into_chunks(row: np.ndarray, n_chunks: int) -> list[np.ndarray]:
    """Split one NaN-padded row into exactly `n_chunks` non-empty chunks."""
    valid = np.isfinite(row)
    row_valid = row[valid]
    true_len = int(len(row_valid))

    if true_len == 0:
        raise ValueError("Row has no finite values after trimming NaN padding")
    if true_len < n_chunks:
        raise ValueError(
            f"true_len={true_len} < n_chunks={n_chunks}; cannot split into non-empty chunks"
        )

    base = true_len // n_chunks
    remainder = true_len % n_chunks

    extra_flags = np.zeros(n_chunks, dtype=int)
    even_positions = list(range(0, n_chunks, 2))
    odd_positions = list(range(1, n_chunks, 2))

    extra_left = remainder
    for pos in even_positions:
        if extra_left <= 0:
            break
        extra_flags[pos] = 1
        extra_left -= 1
    for pos in odd_positions:
        if extra_left <= 0:
            break
        extra_flags[pos] = 1
        extra_left -= 1

    chunks = []
    start = 0
    for j in range(n_chunks):
        length = base + extra_flags[j]
        end = start + length
        chunks.append(row_valid[start:end])
        start = end

    if start != true_len:
        raise ValueError(f"Internal chunking error: consumed {start}, expected {true_len}")

    return chunks
def _chunk_stats(chunk: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (range, mean, std, skewness, excess_kurtosis) for one chunk."""
    m = float(chunk.mean())
    s = float(chunk.std())
    rng = float(chunk.max() - chunk.min())

    if s == 0.0:
        return rng, m, 0.0, 0.0, 0.0

    z = (chunk - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return rng, m, s, skew, kurt
def compute_chunk_feature_map(data_norm: np.ndarray, n_chunks: int) -> dict[str, np.ndarray]:
    """Compute chunk-level features including shape metrics."""
    n_samples, _ = data_norm.shape
    feature_map = {
        "range": np.zeros((n_samples, n_chunks), dtype=float),
        "mean": np.zeros((n_samples, n_chunks), dtype=float),
        "std": np.zeros((n_samples, n_chunks), dtype=float),
        "skewness": np.zeros((n_samples, n_chunks), dtype=float),
        "kurtosis": np.zeros((n_samples, n_chunks), dtype=float),
    }

    for i in range(n_samples):
        chunks = _split_row_into_chunks(data_norm[i], n_chunks)
        for j, chunk in enumerate(chunks):
            rng, m, s, skew, kurt = _chunk_stats(chunk)
            feature_map["range"][i, j] = rng
            feature_map["mean"][i, j] = m
            feature_map["std"][i, j] = s
            feature_map["skewness"][i, j] = skew
            feature_map["kurtosis"][i, j] = kurt

    return feature_map


def compute_shape_shift(
    profiles_3d: np.ndarray,
    group: str = "concat",
) -> np.ndarray:
    """
    Compute day-to-day shape shift for normalized profiles.

    Parameters
    ----------
    profiles_3d : np.ndarray of shape (n_events, n_days, n_points)
        Normalized REM profiles. profiles_3d[e, d, :] is profile for event e, day d.
    group : str
        "concat" — use all n_points.
        "even" — use even-indexed points (0, 2, 4, ...).
        "odd" — use odd-indexed points (1, 3, 5, ...).

    Returns
    -------
    shape_shift : np.ndarray of shape (n_events, n_days - 1)
        shape_shift[e, d] = sum(|profiles_3d[e, d+1] - profiles_3d[e, d]|)
        for d = 0, ..., n_days-2.
    """
    n_points = profiles_3d.shape[-1]

    if group == "concat":
        indices = slice(None)
    elif group == "even":
        indices = slice(0, n_points, 2)
    elif group == "odd":
        indices = slice(1, n_points, 2)
    else:
        raise ValueError(f"Unknown group for shape_shift: {group}")

    selected = profiles_3d[:, :, indices]
    diffs = np.abs(np.diff(selected, axis=1))

    return np.sum(diffs, axis=-1)


def shape_shift_tau_chunk_indices(n_days: int, half_n: int) -> np.ndarray:
    """Map each day-transition index j to the last chunk index of day j."""
    return np.array([(j + 1) * half_n - 1 for j in range(n_days - 1)], dtype=np.int64)


def expected_fixed_n_chunk_count(n_points_per_day: int, n_days: int = FIXED_N_CHUNK_DAYS) -> int:
    """Expected number of chunk columns for fixed-N profiles with concat/even/odd groups."""
    return int(n_days * (n_points_per_day // 2))


def expected_fixed_n_chunks(
    n_points_per_day: int,
    feature_selection: dict[str, list[str]],
    n_days: int = FIXED_N_CHUNK_DAYS,
) -> dict[str, dict[str, int]]:
    """Return the expected number of values per (group, metric)."""
    half_n = n_points_per_day // 2
    n_chunks = n_days * half_n

    result: dict[str, dict[str, int]] = {}
    for group, metrics in feature_selection.items():
        result[group] = {}
        for metric in metrics:
            if metric == _SHAPE_SHIFT_METRIC:
                result[group][metric] = n_days - 1
            else:
                result[group][metric] = n_chunks
    return result


def compute_chunk_feature_map_fixed_n(
    profiles: np.ndarray,
    n_points_per_day: int,
    feature_selection: dict[str, list[str]],
    n_days: int = FIXED_N_CHUNK_DAYS,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute chunk features with per-day normalization for fixed-N REM profiles.

    Parameters
    ----------
    profiles : np.ndarray of shape (n_events, n_days * n_points_per_day)
        Concatenated daily profiles (may include extra days; first ``n_days`` are used).
    n_points_per_day : int
        Number of profile points per day.
    feature_selection : dict
        e.g., {"concat": ["mean", "range"], "even": ["mean"], "odd": ["shape_shift"]}
    n_days : int
        Number of days in the chunking window (default 8).

    Returns
    -------
    group_data : dict
        {group_name: {metric_name: np.ndarray of shape (n_events, n_chunks)}}
    """
    n_events = profiles.shape[0]
    half_n = n_points_per_day // 2
    total_points = n_days * n_points_per_day
    if profiles.shape[1] < total_points:
        raise ValueError(
            f"profiles width {profiles.shape[1]} < required {total_points} "
            f"for n_days={n_days}, n_points_per_day={n_points_per_day}"
        )

    profiles_slice = profiles[:, :total_points]
    profiles_3d = profiles_slice.reshape(n_events, n_days, n_points_per_day)

    profiles_norm = np.zeros_like(profiles_3d)
    for d in range(n_days):
        day_data = profiles_3d[:, d, :]
        day_min = day_data.min(axis=1, keepdims=True)
        day_max = day_data.max(axis=1, keepdims=True)
        denom = day_max - day_min
        denom[denom == 0] = 1.0
        profiles_norm[:, d, :] = (day_data - day_min) / denom

    group_data: dict[str, dict[str, np.ndarray]] = {}

    for group_name, metrics in feature_selection.items():
        group_data[group_name] = {}

        if _SHAPE_SHIFT_METRIC in metrics:
            ss_values = compute_shape_shift(profiles_norm, group=group_name)
            group_data[group_name][_SHAPE_SHIFT_METRIC] = ss_values

        chunk_metrics = [m for m in metrics if m != _SHAPE_SHIFT_METRIC]
        if not chunk_metrics:
            continue

        if group_name == "concat":
            chunks = profiles_norm.reshape(n_events, n_days, half_n, 2)
        elif group_name == "even":
            chunks = profiles_norm[:, :, 0::2].reshape(n_events, n_days, half_n, 1)
        elif group_name == "odd":
            chunks = profiles_norm[:, :, 1::2].reshape(n_events, n_days, half_n, 1)
        else:
            raise ValueError(f"Unknown group: {group_name}")

        for metric in chunk_metrics:
            if metric == "mean":
                values = chunks.mean(axis=-1)
            elif metric == "range":
                values = chunks.max(axis=-1) - chunks.min(axis=-1)
            elif metric == "std":
                values = chunks.std(axis=-1, ddof=1)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            group_data[group_name][metric] = values.reshape(n_events, -1)

    return group_data


def compute_concat_chunk_feature_map(data_norm: np.ndarray, n_chunks: int) -> dict[str, np.ndarray]:
    """Compute features on concatenated odd+even chunk pairs."""
    if n_chunks % 2 != 0:
        raise ValueError("concat group requires even n_chunks so odd/even pairs are complete")

    n_samples, _ = data_norm.shape
    n_concat = n_chunks // 2
    feature_map = {
        "range": np.zeros((n_samples, n_concat), dtype=float),
        "mean": np.zeros((n_samples, n_concat), dtype=float),
        "std": np.zeros((n_samples, n_concat), dtype=float),
        "skewness": np.zeros((n_samples, n_concat), dtype=float),
        "kurtosis": np.zeros((n_samples, n_concat), dtype=float),
    }

    for i in range(n_samples):
        chunks = _split_row_into_chunks(data_norm[i], n_chunks)
        for k in range(n_concat):
            merged = np.concatenate([chunks[2 * k], chunks[2 * k + 1]])
            rng, m, s, skew, kurt = _chunk_stats(merged)
            feature_map["range"][i, k] = rng
            feature_map["mean"][i, k] = m
            feature_map["std"][i, k] = s
            feature_map["skewness"][i, k] = skew
            feature_map["kurtosis"][i, k] = kurt

    return feature_map
def compute_chunk_features(data_norm: np.ndarray, n_chunks: int):
    """Backward-compatible helper: returns only (range, mean)."""
    features = compute_chunk_feature_map(data_norm, n_chunks=n_chunks)
    return features["range"], features["mean"]
def _to_feature_df(values: np.ndarray, prefix: str) -> pd.DataFrame:
    n_cols = values.shape[1]
    columns = [f"{prefix}_chunk_{i+1}" for i in range(n_cols)]
    return pd.DataFrame(values, columns=columns)
def _selection_uses_shape_shift(selection: dict[str, list[str]]) -> bool:
    return any(_SHAPE_SHIFT_METRIC in feats for feats in selection.values())


def _validate_shape_shift_groups(selection: dict[str, list[str]]) -> dict[str, list[str]]:
    shape_groups = [
        group_name
        for group_name, feats in selection.items()
        if _SHAPE_SHIFT_METRIC in feats
    ]
    for group_name in shape_groups:
        if group_name not in _SHAPE_SHIFT_GROUPS:
            raise ValueError(
                f"Metric '{_SHAPE_SHIFT_METRIC}' is not supported for group '{group_name}'. "
                f"Use one of: {sorted(_SHAPE_SHIFT_GROUPS)}."
            )
    return selection


def _parse_feature_selection(feature_selection) -> dict[str, list[str]]:
    base_metrics = {"range", "mean", "std", "skewness", "kurtosis", _SHAPE_SHIFT_METRIC}
    base_groups = {"all", "odd", "even", "concat", "single"}

    def norm_group(g: str) -> str:
        g = g.strip().lower()
        if g == "single":
            return "all"
        return g

    out: dict[str, list[str]] = {}

    if isinstance(feature_selection, dict):
        for group_name, feats in feature_selection.items():
            g = norm_group(str(group_name))
            if g not in {"all", "odd", "even", "concat"}:
                raise ValueError(f"Unknown group '{group_name}'. Use one of: all, odd, even, concat")

            feats_list = [feats] if isinstance(feats, str) else list(feats)
            out[g] = []
            for feat in feats_list:
                f = str(feat).strip().lower()
                if f not in base_metrics:
                    raise ValueError(f"Unknown metric '{feat}'. Use one of: {sorted(base_metrics)}")
                if f not in out[g]:
                    out[g].append(f)
        return _validate_shape_shift_groups(out)

    if isinstance(feature_selection, (list, tuple, set)):
        for item in feature_selection:
            txt = str(item).strip().lower().replace("_", " ")
            parts = [p for p in txt.split() if p]
            if len(parts) != 2:
                raise ValueError(
                    f"Cannot parse '{item}'. Expected two tokens like 'mean even' or 'even mean'."
                )

            a, b = parts
            if a in base_metrics and b in base_groups:
                f, g = a, norm_group(b)
            elif b in base_metrics and a in base_groups:
                f, g = b, norm_group(a)
            else:
                raise ValueError(
                    f"Cannot parse '{item}'. Need one metric and one group token."
                )

            if g not in out:
                out[g] = []
            if f not in out[g]:
                out[g].append(f)
        return _validate_shape_shift_groups(out)

    raise ValueError(
        "feature_selection must be dict or list/tuple/set of strings, e.g. "
        "{'even':['mean','range'], 'odd':['kurtosis','skewness']} or "
        "['mean even','range even','kurtosis odd','skewness odd']."
    )
def _day_lengths_for_shape_shift(
    data_raw: np.ndarray,
    *,
    day_lengths_per_sample: list[list[int]] | None = None,
    csv_path: str | None = None,
) -> list[list[int]] | None:
    """Resolve per-event day profile lengths for shape_shift computation."""
    if day_lengths_per_sample is not None:
        return day_lengths_per_sample

    prep_cfg = get_runtime_prepare_cfg() or {}
    lengths = prep_cfg.get("day_lengths_per_sample")
    if lengths is not None:
        return lengths

    csv_path = csv_path or prep_cfg.get("csv_path")
    if csv_path and os.path.isfile(csv_path):
        return _resolve_day_lengths_per_sample(
            csv_path,
            data_raw.shape[0],
            export_cfg=get_runtime_export_cfg(),
        )

    export_cfg = get_runtime_export_cfg() or {}
    out_dir = export_cfg.get("output_dir")
    nanpad = export_cfg.get("output_csv_nanpad")
    if out_dir and nanpad:
        candidate = os.path.join(str(out_dir), str(nanpad))
        if os.path.isfile(candidate):
            return _resolve_day_lengths_per_sample(
                candidate,
                data_raw.shape[0],
                export_cfg=export_cfg,
            )
    return None


def build_group_data(
    data_norm: np.ndarray,
    *,
    n_chunks: int,
    feature_selection,
    data_raw: np.ndarray | None = None,
    window_days: int | None = None,
    shape_shift_fill_first: bool = False,
    day_lengths_per_sample: list[list[int]] | None = None,
    csv_path: str | None = None,
    n_points_per_day: int | None = None,
    fixed_n_days: int = FIXED_N_CHUNK_DAYS,
) -> dict:
    """Build group_data dict used by changepoint model from feature_selection only."""

    selection = _parse_feature_selection(feature_selection)
    uses_shape_shift = _selection_uses_shape_shift(selection)

    export_cfg = get_runtime_export_cfg() or {}
    if n_points_per_day is None and export_cfg.get("n_points_per_day") is not None:
        n_points_per_day = int(export_cfg["n_points_per_day"])
    fixed_n_mode = n_points_per_day is not None

    if uses_shape_shift and not fixed_n_mode:
        raise NotImplementedError(
            "shape_shift requires fixed-N mode (set n_points_per_day or export_cfg n_points_per_day)."
        )

    chunk_metrics_needed = any(
        metric_name != _SHAPE_SHIFT_METRIC
        for metric_list in selection.values()
        for metric_name in metric_list
    )

    fixed_maps: dict[str, dict[str, np.ndarray]] | None = None
    feature_map: dict[str, np.ndarray] = {}
    concat_feature_map: dict[str, np.ndarray] | None = None

    if fixed_n_mode and (chunk_metrics_needed or uses_shape_shift):
        if data_raw is None:
            raise ValueError("data_raw is required for fixed-N chunk features.")
        n_pts = int(n_points_per_day)
        expected_chunk = expected_fixed_n_chunk_count(n_pts, fixed_n_days)
        expected_shape = fixed_n_days - 1

        if chunk_metrics_needed and uses_shape_shift:
            if int(n_chunks) != expected_chunk:
                raise ValueError(
                    f"fixed-N joint mode with n_points_per_day={n_pts}, n_days={fixed_n_days} "
                    f"requires n_chunks={expected_chunk}, got {n_chunks}."
                )
        elif chunk_metrics_needed:
            if int(n_chunks) != expected_chunk:
                raise ValueError(
                    f"fixed-N mode with n_points_per_day={n_pts}, n_days={fixed_n_days} "
                    f"requires n_chunks={expected_chunk}, got {n_chunks}."
                )
        else:
            if int(n_chunks) != expected_shape:
                raise ValueError(
                    f"fixed-N shape_shift-only mode with n_days={fixed_n_days} "
                    f"requires n_chunks={expected_shape}, got {n_chunks}."
                )

        fixed_selection = {
            group_key: list(metric_list)
            for group_key, metric_list in selection.items()
            if group_key in {"concat", "odd", "even"}
        }
        fixed_maps = compute_chunk_feature_map_fixed_n(
            data_raw,
            n_pts,
            fixed_selection,
            n_days=fixed_n_days,
        )
    elif chunk_metrics_needed:
        feature_map = compute_chunk_feature_map(data_norm, n_chunks=n_chunks)
        if any(group_key == "concat" for group_key in selection):
            concat_feature_map = compute_concat_chunk_feature_map(data_norm, n_chunks=n_chunks)

    idx_map = {
        "all": np.arange(n_chunks),
        "odd": np.arange(0, n_chunks, 2),
        "even": np.arange(1, n_chunks, 2),
        "concat": np.arange(n_chunks // 2),
    }

    feature_source_map = {
        "all": feature_map,
        "odd": feature_map,
        "even": feature_map,
        "concat": concat_feature_map if concat_feature_map is not None else feature_map,
    }

    def _validate_same_chunk_count(out_dict: dict) -> None:
        chunk_counts: set[int] = set()
        for feats in out_dict.values():
            for metric_name, df in feats.items():
                if metric_name == _SHAPE_SHIFT_METRIC:
                    continue
                chunk_counts.add(df.shape[1])
        if len(chunk_counts) > 1:
            raise ValueError(
                "Selected groups produce different chunk counts, incompatible with shared tau. "
                f"Chunk counts found: {sorted(chunk_counts)}."
            )

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for group_key, metric_list in selection.items():
        if group_key not in idx_map and group_key != "concat":
            raise ValueError(f"Unknown group '{group_key}' after parsing")
        out[group_key] = {}
        for metric_name in metric_list:
            prefix = metric_name if group_key == "all" else f"{metric_name}_{group_key}"
            if fixed_n_mode:
                if fixed_maps is None:
                    raise ValueError("Internal error: fixed_maps not computed.")
                if group_key not in fixed_maps:
                    raise ValueError(
                        f"Group '{group_key}' not available in fixed-N feature maps."
                    )
                if metric_name not in fixed_maps[group_key]:
                    raise ValueError(
                        f"Metric '{metric_name}' not available for group '{group_key}'."
                    )
                out[group_key][metric_name] = _to_feature_df(
                    fixed_maps[group_key][metric_name],
                    prefix,
                )
                continue
            src = feature_source_map[group_key]
            if metric_name not in src:
                raise ValueError(
                    f"Metric '{metric_name}' is not available for group '{group_key}'."
                )
            out[group_key][metric_name] = _to_feature_df(
                src[metric_name][:, idx_map[group_key]],
                prefix,
            )

    _validate_same_chunk_count(out)
    return out
def _chunk_group_idx_map(n_chunks: int) -> dict[str, np.ndarray]:
    return {
        "all": np.arange(n_chunks),
        "odd": np.arange(0, n_chunks, 2),
        "even": np.arange(1, n_chunks, 2),
        "concat": np.arange(n_chunks // 2),
    }
def _rem_profile_key_tuple(rem_profile_params: dict | None) -> tuple:
    rpp = rem_profile_params or {}
    if "n_points_per_day" in rpp:
        return (
            int(rpp["n_points_per_day"]),
            float(rpp.get("overlap", 0.0)),
            int(rpp.get("rem_stage", 2)),
        )
    return (
        int(rpp.get("window_size_hours", 0)),
        int(rpp.get("step_size_hours", 0)),
    )
def precompute_all_features(data_norm: np.ndarray, config: dict) -> dict[tuple, np.ndarray]:
    """Precompute chunk features for all proposal-grid combinations.

    Returns a dict keyed by ``(window_size, step_size, n_chunks, group, metric)``
    mapping to arrays of shape ``(n_events, n_chunks_for_group)``.
    """
    proposal_options = config.get("proposal_options", config)
    rem_choices = list(proposal_options.get("rem_profile_choices") or [])
    n_choices = list(proposal_options.get("n_chunks_choices") or [])
    groups = list(proposal_options.get("allowed_groups") or ["concat", "odd", "even", "all"])
    metrics = list(proposal_options.get("allowed_metrics") or ["mean", "range"])

    if not n_choices:
        raise ValueError("precompute_all_features requires n_chunks_choices in proposal_options")

    export_cfg_base = get_runtime_export_cfg()
    prep_cfg = get_runtime_prepare_cfg() or {}
    fallback_rem = config.get("rem_profile_params")
    rem_profiles: list[dict | None]
    if rem_choices:
        rem_profiles = [dict(r) for r in rem_choices]
    elif fallback_rem:
        rem_profiles = [dict(fallback_rem)]
    else:
        rem_profiles = [None]

    precomputed: dict[tuple, np.ndarray] = {}
    for rem in rem_profiles:
        draw: np.ndarray | None = None
        window_days = int((export_cfg_base or {}).get("window_days", 10))
        rem_validated: dict | None = None
        if rem and export_cfg_base:
            from seismic_pipeline.config.changepoint_defaults import validate_rem_profile_params
            from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only

            rem_validated = validate_rem_profile_params(rem)
            export_cfg = dict(export_cfg_base)
            export_cfg.update(
                {
                    "n_points_per_day": int(rem_validated["n_points_per_day"]),
                    "overlap": float(rem_validated["overlap"]),
                    "rem_stage": int(rem_validated["rem_stage"]),
                }
            )
            window_days = int(export_cfg.get("window_days", window_days))
            export_result = export_rem_profiles_10days_cached_only(**export_cfg)
            prep = prepare_model_data(
                csv_path=export_result["paths"]["nanpad_output_csv"],
                bad_sample_indices=prep_cfg.get("bad_sample_indices"),
            )
            dnorm = prep["data_norm"]
            draw = prep["data_raw"]
        else:
            dnorm = data_norm
            draw = get_runtime_data_raw()
            if rem:
                from seismic_pipeline.config.changepoint_defaults import validate_rem_profile_params
                rem_validated = validate_rem_profile_params(rem)

        rem_key = _rem_profile_key_tuple(rem_validated if rem_validated else (rem if rem else fallback_rem))
        fixed_n_mode = rem_validated is not None and "n_points_per_day" in rem_validated

        for n_chunks in n_choices:
            n_chunks = int(n_chunks)
            chunk_metrics = [m for m in metrics if m != _SHAPE_SHIFT_METRIC]
            shape_shift_needed = _SHAPE_SHIFT_METRIC in metrics
            if chunk_metrics or shape_shift_needed:
                if fixed_n_mode and draw is not None:
                    n_pts = int(rem_validated["n_points_per_day"])
                    fixed_selection: dict[str, list[str]] = {}
                    for g in groups:
                        if g not in {"concat", "odd", "even"}:
                            continue
                        g_metrics = [m for m in metrics if m in chunk_metrics or m == _SHAPE_SHIFT_METRIC]
                        if g_metrics:
                            fixed_selection[g] = g_metrics
                    if fixed_selection:
                        fixed_maps = compute_chunk_feature_map_fixed_n(
                            draw,
                            n_pts,
                            fixed_selection,
                            n_days=FIXED_N_CHUNK_DAYS,
                        )
                        for group in groups:
                            if group not in fixed_maps:
                                continue
                            for metric in fixed_selection.get(group, []):
                                if metric not in fixed_maps[group]:
                                    continue
                                key = (*rem_key, n_chunks, group, metric)
                                precomputed[key] = fixed_maps[group][metric].copy()
                elif chunk_metrics:
                    feature_map = compute_chunk_feature_map(dnorm, n_chunks=n_chunks)
                    concat_feature_map = None
                    if "concat" in groups and any(m in feature_map for m in chunk_metrics):
                        concat_feature_map = compute_concat_chunk_feature_map(dnorm, n_chunks=n_chunks)
                    idx_map = _chunk_group_idx_map(n_chunks)
                    feature_source_map = {
                        "all": feature_map,
                        "odd": feature_map,
                        "even": feature_map,
                        "concat": concat_feature_map,
                    }

                    for group in groups:
                        if group not in idx_map:
                            continue
                        src = feature_source_map.get(group)
                        if src is None:
                            continue
                        for metric in chunk_metrics:
                            if metric not in src:
                                continue
                            key = (*rem_key, n_chunks, group, metric)
                            precomputed[key] = src[metric][:, idx_map[group]].copy()

    return precomputed
def group_data_from_precomputed(
    precomputed: dict[tuple, np.ndarray],
    config: dict,
) -> dict:
    """Build group_data from a precomputed feature dict (same layout as build_group_data)."""
    n_chunks = int(config["n_chunks"])
    rem_key = _rem_profile_key_tuple(config.get("rem_profile_params"))
    selection = _parse_feature_selection(config["feature_selection"])
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for group_key, metric_list in selection.items():
        out[group_key] = {}
        for metric_name in metric_list:
            key = (*rem_key, n_chunks, group_key, metric_name)
            if key not in precomputed:
                raise KeyError(
                    f"Precomputed features missing for key={key!r}. "
                    "Check rem_profile_choices, n_chunks_choices, allowed_groups, and allowed_metrics."
                )
            prefix = metric_name if group_key == "all" else f"{metric_name}_{group_key}"
            out[group_key][metric_name] = _to_feature_df(precomputed[key], prefix)

    counts: set[int] = set()
    shape_shift_counts: set[int] = set()
    for feats in out.values():
        for metric_name, df in feats.items():
            if metric_name == _SHAPE_SHIFT_METRIC:
                shape_shift_counts.add(df.shape[1])
            else:
                counts.add(df.shape[1])
    if len(counts) > 1:
        raise ValueError(
            "Selected groups produce different chunk counts, incompatible with shared tau. "
            f"Chunk counts found: {sorted(counts)}."
        )
    return out
def prepare_variant_data(data_norm: np.ndarray, mode: str):
    """Backward-compatible scenario helper returning grouped range/mean DataFrames."""
    if mode == "ten_unsplit":
        range_all, mean_all = compute_chunk_features(data_norm, n_chunks=10)
        return {
            "all": {
                "range": _to_feature_df(range_all, "range"),
                "mean": _to_feature_df(mean_all, "mean"),
            }
        }

    if mode == "twenty_even_odd":
        range_20, mean_20 = compute_chunk_features(data_norm, n_chunks=20)
        odd_idx = np.arange(0, 20, 2)
        even_idx = np.arange(1, 20, 2)
        return {
            "odd": {
                "range": _to_feature_df(range_20[:, odd_idx], "range_odd"),
                "mean": _to_feature_df(mean_20[:, odd_idx], "mean_odd"),
            },
            "even": {
                "range": _to_feature_df(range_20[:, even_idx], "range_even"),
                "mean": _to_feature_df(mean_20[:, even_idx], "mean_even"),
            },
        }

    if mode == "twenty_even_only":
        range_20, mean_20 = compute_chunk_features(data_norm, n_chunks=20)
        even_idx = np.arange(1, 20, 2)
        return {
            "even": {
                "range": _to_feature_df(range_20[:, even_idx], "range_even"),
                "mean": _to_feature_df(mean_20[:, even_idx], "mean_even"),
            }
        }

    raise ValueError(f"Неизвестный режим: {mode}")
def prepare_model_data(
    csv_path: str = "samples_10days_nanpad.csv",
    bad_sample_indices: List[int] | None = None,
) -> dict:
    """Load/normalize CSV and apply sample exclusion mask for modeling."""
    global _RUNTIME_LAST_PREPARE_CFG

    data_raw, data_norm = load_and_normalize(csv_path)
    day_lengths_all = _resolve_day_lengths_per_sample(
        csv_path,
        n_samples=data_raw.shape[0],
        export_cfg=get_runtime_export_cfg(),
    )
    data_raw_plot = data_raw.copy()
    data_norm_plot = data_norm.copy()

    bad_sample_indices = bad_sample_indices or []
    n_samples = data_norm_plot.shape[0]
    bad_set = set(bad_sample_indices)
    for idx in bad_set:
        if idx < 0 or idx >= n_samples:
            raise ValueError(f"bad_sample_indices contains out-of-range idx={idx}, n_samples={n_samples}")

    good_indices = np.array([i for i in range(n_samples) if i not in bad_set], dtype=int)
    if good_indices.size == 0:
        raise ValueError("All samples are excluded from model fitting; check bad_sample_indices.")

    data_raw_model = data_raw_plot[good_indices]
    data_norm_model = data_norm_plot[good_indices]
    day_lengths_model = (
        [day_lengths_all[int(i)] for i in good_indices.tolist()]
        if day_lengths_all is not None
        else None
    )
    _RUNTIME_LAST_PREPARE_CFG = {
        "csv_path": csv_path,
        "bad_sample_indices": sorted(bad_set),
        "day_lengths_per_sample": day_lengths_model,
    }
    set_runtime_prepare_cfg(_RUNTIME_LAST_PREPARE_CFG)
    set_runtime_data_raw(data_raw_model)
    return {
        "data_raw": data_raw_model,
        "data_norm": data_norm_model,
        "data_raw_plot": data_raw_plot,
        "data_norm_plot": data_norm_plot,
        "good_indices": good_indices,
        "bad_set": bad_set,
        "n_samples_original": n_samples,
        "day_lengths_per_sample": day_lengths_model,
        "csv_path": csv_path,
    }
