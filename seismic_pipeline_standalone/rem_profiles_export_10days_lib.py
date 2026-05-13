"""
Cached-only 10-day REM profile export utilities.

This module provides a notebook- and CLI-friendly API for exporting samples from
already available hypnograms (cache/local/S3 temp). It does not include any
quality-model or auto-hypnogram-from-dat logic.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

from seismic_pipeline import HypnogramCacheManagerYt, REMProfileCalculatorYt


DEFAULT_EVENTS_10D: List[Dict[str, str]] = [
    {"rat_id": "R2", "date": "2022-11-07"},
    {"rat_id": "R2", "date": "2022-11-18"},
    {"rat_id": "R2", "date": "2023-04-03"},
    # {"rat_id": "R2", "date": "2023-04-11"},
    # {"rat_id": "R2", "date": "2023-04-18"},
    # {"rat_id": "R2", "date": "2023-04-21"},
    {"rat_id": "R2", "date": "2023-05-03"},
    # {"rat_id": "R2", "date": "2023-05-09"},
    {"rat_id": "R2", "date": "2024-09-30"},
    {"rat_id": "R2", "date": "2024-10-29"},
    {"rat_id": "R3", "date": "2025-01-23"},
    {"rat_id": "R3", "date": "2025-03-14"},
    {"rat_id": "R1", "date": "2025-07-02"},
    {"rat_id": "R2", "date": "2025-07-02"},
    {"rat_id": "R3", "date": "2025-07-02"},
    {"rat_id": "R4", "date": "2025-07-02"},
    {"rat_id": "R1", "date": "2025-07-20"},
    {"rat_id": "R2", "date": "2025-07-20"},
    {"rat_id": "R3", "date": "2025-07-20"},
    {"rat_id": "R4", "date": "2025-07-20"},
]


DEFAULT_S3_CONFIG = {
    "service_name": "s3",
    "endpoint_url": "http://10.132.230.2:7770",
    "aws_access_key_id": "quantum",
    "aws_secret_access_key": "s3password",
}


def _normalize_rem_profile_params(
    *,
    window_size_hours: int,
    step_size_hours: int,
    rem_stage: int,
) -> tuple[int, int, int]:
    """Validate and normalize REM profile generation parameters."""
    raw = {
        "window_size_hours": window_size_hours,
        "step_size_hours": step_size_hours,
        "rem_stage": rem_stage,
    }
    coerced: dict[str, int] = {}
    for name, value in raw.items():
        try:
            coerced[name] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc

    w, s, r = coerced["window_size_hours"], coerced["step_size_hours"], coerced["rem_stage"]
    if w <= 0:
        raise ValueError(f"window_size_hours must be > 0, got {w}")
    if s <= 0:
        raise ValueError(f"step_size_hours must be > 0, got {s}")
    if r < 0:
        raise ValueError(f"rem_stage must be >= 0, got {r}")

    return w, s, r


def _parse_event_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _build_10day_inputs(events: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for event in events:
        dt = _parse_event_date(event["date"])
        raw_direction = str(event.get("direction", "before")).strip().lower()
        if raw_direction == "after":
            # Backward-compatible alias for "after_reversed".
            direction = "after_reversed"
        elif raw_direction in {"before", "after_reversed"}:
            direction = raw_direction
        else:
            raise ValueError(
                f"Unsupported direction '{event.get('direction')}'. "
                "Use 'before' or 'after_reversed'."
            )

        if direction == "before":
            window_dates = [
                (dt - timedelta(days=offset)).strftime("%Y_%m_%d")
                for offset in range(10, 0, -1)
            ]
        else:
            # Reverse order for post-event window: +10, +9, ... +1.
            window_dates = [
                (dt + timedelta(days=offset)).strftime("%Y_%m_%d")
                for offset in range(10, 0, -1)
            ]
        out.append(
            {
                "rat_id": event["rat_id"],
                "window_dates": window_dates,
                "original_event_date": event["date"],
                "original_rat_id": event["rat_id"],
                "window_direction": direction,
            }
        )
    return out


def _cache_needed_dates(
    cache_manager: HypnogramCacheManagerYt,
    rows: List[Dict[str, object]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    needed = set()
    for row in rows:
        rat_id = row["rat_id"]
        for date in row["window_dates"]:
            needed.add((rat_id, date))

    cached: List[Tuple[str, str]] = []
    missing: List[Tuple[str, str]] = []
    for rat_id, date in sorted(needed):
        if cache_manager.get_cached_hypnogram(rat_id, date) is not None:
            cached.append((rat_id, date))
            continue
        if cache_manager.cache_hypnogram(rat_id, date, source="local"):
            cached.append((rat_id, date))
            continue
        if cache_manager.cache_hypnogram(rat_id, date, source="s3"):
            cached.append((rat_id, date))
            continue
        missing.append((rat_id, date))
        cache_manager.mark_missing(rat_id, date)
    return cached, missing


def _extract_hypnogram_array(hypnogram_obj) -> np.ndarray | None:
    hypnogram = hypnogram_obj
    if isinstance(hypnogram, list):
        hypnogram = hypnogram[0] if hypnogram else None
    if hypnogram is None:
        return None
    arr = np.asarray(hypnogram)
    if arr.size == 0:
        return None
    return arr


def _build_concat_features(
    export_rows: List[Dict[str, object]],
    cache_manager: HypnogramCacheManagerYt,
    rem_calc: REMProfileCalculatorYt,
    require_full_window_for_concat: bool,
) -> Tuple[List[np.ndarray], List[int]]:
    raw_features: List[np.ndarray] = []
    valid_indices: List[int] = []

    for i, row in enumerate(export_rows):
        rat_id = row["rat_id"]
        window_dates = row["window_dates"]
        hypno_parts: List[np.ndarray] = []

        for date in window_dates:
            hypnogram_obj = cache_manager.get_cached_hypnogram(rat_id, date)
            hyp_arr = _extract_hypnogram_array(hypnogram_obj)
            if hyp_arr is not None:
                hypno_parts.append(hyp_arr)

        if require_full_window_for_concat and len(hypno_parts) != len(window_dates):
            continue
        if not hypno_parts:
            continue

        big_hypno = np.concatenate(hypno_parts)
        rem_profile = rem_calc._fraction(
            big_hypno,
            (rem_calc.window_size_hours, rem_calc.step_size_hours),
            rem_calc.rem_stage,
            rem_calc.epoch_length_sec,
        )
        raw_features.append(np.asarray(rem_profile, dtype=float))
        valid_indices.append(i)

    return raw_features, valid_indices


def _build_metadata(
    rows: List[Dict[str, object]],
    missing_pairs: List[Tuple[str, str]],
    exported_original_indices: np.ndarray,
    true_vector_lengths: List[int],
    padded_vector_length: int,
) -> pd.DataFrame:
    missing_set = set(missing_pairs)
    records = []
    exported_set = set(int(i) for i in exported_original_indices.tolist())

    for idx, row in enumerate(rows):
        event_missing = [
            d for d in row["window_dates"] if (row["rat_id"], d) in missing_set
        ]
        exported = idx in exported_set
        true_len = int(true_vector_lengths[idx]) if idx < len(true_vector_lengths) else 0
        pad_count = int(padded_vector_length - true_len) if exported else 0
        records.append(
            {
                "row_index": idx,
                "rat_id": row["rat_id"],
                "event_date": row["original_event_date"],
                "window_direction": row.get("window_direction", "before"),
                "window_dates": ";".join(row["window_dates"]),
                "missing_dates": ";".join(event_missing),
                "missing_count": len(event_missing),
                "exported": exported,
                "vector_length": int(padded_vector_length) if exported else 0,
                "true_vector_length": true_len,
                "pad_count": pad_count,
            }
        )
    return pd.DataFrame(records)


def export_rem_profiles_10days_cached_only(
    *,
    events: List[Dict[str, str]],
    output_dir: str = ".",
    output_csv: str = "samples_10days.csv",
    output_csv_nanpad: str = "samples_10days_nanpad.csv",
    metadata_csv: str = "samples_10days_metadata.csv",
    local_data_root: str = "/mnt/wd/rat",
    local_hypnogram_cache_dir: str = "./hypnogram_cache_legacy10",
    window_size_hours: int = 6,
    step_size_hours: int = 1,
    rem_stage: int = 2,
    epoch_length_sec: int = 5,
    sampling_rate: int = 250,
    concat_hypnogram_for_event: bool = False,
    require_full_window_for_concat: bool = True,
    drop_incomplete_events: bool = True,
    s3_config: Dict[str, str] | None = None,
    s3_rat_bucket: str = "rat",
    s3_temp_bucket: str = "temp",
) -> dict:
    """Export REM profile vectors from cached/local/S3 hypnograms only."""
    global _RUNTIME_LAST_EXPORT_CFG

    window_size_hours, step_size_hours, rem_stage = _normalize_rem_profile_params(
        window_size_hours=window_size_hours,
        step_size_hours=step_size_hours,
        rem_stage=rem_stage,
    )
    _RUNTIME_LAST_EXPORT_CFG = {
        "events": list(events),
        "output_dir": output_dir,
        "output_csv": output_csv,
        "output_csv_nanpad": output_csv_nanpad,
        "metadata_csv": metadata_csv,
        "local_data_root": local_data_root,
        "local_hypnogram_cache_dir": local_hypnogram_cache_dir,
        "window_size_hours": window_size_hours,
        "step_size_hours": step_size_hours,
        "rem_stage": rem_stage,
        "epoch_length_sec": epoch_length_sec,
        "sampling_rate": sampling_rate,
        "concat_hypnogram_for_event": concat_hypnogram_for_event,
        "require_full_window_for_concat": require_full_window_for_concat,
        "drop_incomplete_events": drop_incomplete_events,
        "s3_config": s3_config,
        "s3_rat_bucket": s3_rat_bucket,
        "s3_temp_bucket": s3_temp_bucket,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, output_csv)
    output_csv_nanpad_path = os.path.join(output_dir, output_csv_nanpad)
    metadata_csv_path = os.path.join(output_dir, metadata_csv)
    summary_json_path = os.path.join(output_dir, "samples_10days_summary.json")

    cache_manager = HypnogramCacheManagerYt(
        local_cache_dir=local_hypnogram_cache_dir,
        local_data_root=local_data_root,
        s3_config=s3_config or DEFAULT_S3_CONFIG,
        s3_rat_bucket=s3_rat_bucket,
        s3_temp_bucket=s3_temp_bucket,
        # Keep fallback roots enabled so notebook configs with '/mnt/wd/rat'
        # still resolve to '~/mnt/wd/rat' when mounted there.
        allow_local_root_fallback=True,
    )

    rows = _build_10day_inputs(events)
    cached_pairs, missing_pairs = _cache_needed_dates(cache_manager, rows)
    print(f"Total required (rat,date) pairs: {len(cached_pairs) + len(missing_pairs)}")
    print(f"Initially cached/available: {len(cached_pairs)}")
    print(f"Initially missing: {len(missing_pairs)}")
    print(
        "REM profile params: "
        f"window_size_hours={window_size_hours}, "
        f"step_size_hours={step_size_hours}, "
        f"rem_stage={rem_stage}"
    )

    missing_set = set(missing_pairs)
    if drop_incomplete_events:
        export_rows = []
        kept_indices = []
        for idx, row in enumerate(rows):
            has_missing = any((row["rat_id"], d) in missing_set for d in row["window_dates"])
            if not has_missing:
                export_rows.append(row)
                kept_indices.append(idx)
        kept_indices_array = np.array(kept_indices, dtype=int)
    else:
        export_rows = rows
        kept_indices_array = np.arange(len(rows), dtype=int)

    rem_calc = REMProfileCalculatorYt(
        cache_manager=cache_manager,
        window_size_hours=window_size_hours,
        step_size_hours=step_size_hours,
        rem_stage=rem_stage,
        epoch_length_sec=epoch_length_sec,
        sampling_rate=sampling_rate,
        fail_on_missing_data=False,
    )

    if concat_hypnogram_for_event:
        raw_features, valid_indices = _build_concat_features(
            export_rows=export_rows,
            cache_manager=cache_manager,
            rem_calc=rem_calc,
            require_full_window_for_concat=require_full_window_for_concat,
        )
    else:
        raw_features, valid_indices = rem_calc._calculate_features_for_X(export_rows)

    if not raw_features:
        raise ValueError(
            "No REM profile vectors were produced from the selected rows. "
            "Check missing data and cache contents."
        )

    padded_vector_length = int(max(len(v) for v in raw_features))
    exported_original_indices: List[int] = []
    true_vector_lengths: List[int] = [0] * len(rows)
    padded_zero_rows: List[np.ndarray] = []
    padded_nan_rows: List[np.ndarray] = []

    for k, features in enumerate(raw_features):
        export_row_idx = int(valid_indices[k])
        original_row_idx = int(kept_indices_array[export_row_idx])
        true_len = int(len(features))
        true_vector_lengths[original_row_idx] = true_len
        exported_original_indices.append(original_row_idx)

        pad_width = padded_vector_length - true_len
        if pad_width < 0:
            features_padded_zero = features[:padded_vector_length]
            features_padded_nan = features[:padded_vector_length]
        else:
            features_padded_zero = np.pad(features, (0, pad_width), "constant", constant_values=0.0)
            features_padded_nan = np.pad(features, (0, pad_width), "constant", constant_values=np.nan)

        padded_zero_rows.append(features_padded_zero)
        padded_nan_rows.append(features_padded_nan)

    X_zero = np.vstack(padded_zero_rows)
    X_nan = np.vstack(padded_nan_rows)

    pd.DataFrame(X_zero).to_csv(output_csv_path, index=False, header=False)
    pd.DataFrame(X_nan).to_csv(output_csv_nanpad_path, index=False, header=False)

    metadata_df = _build_metadata(
        rows=rows,
        missing_pairs=missing_pairs,
        exported_original_indices=np.array(exported_original_indices, dtype=int),
        true_vector_lengths=true_vector_lengths,
        padded_vector_length=padded_vector_length,
    )
    metadata_df.to_csv(metadata_csv_path, index=False)

    non_empty_exported = [idx for idx in exported_original_indices if true_vector_lengths[idx] > 0]
    summary = {
        "events_total": len(rows),
        "events_exported": int(len(exported_original_indices)),
        "required_pairs_total": len(cached_pairs) + len(missing_pairs),
        "missing_pairs_final": len(missing_pairs),
        "matrix_shape": [int(len(exported_original_indices)), int(padded_vector_length)],
        "stable_vector_length": bool(padded_vector_length > 0),
        "row_non_empty": bool(non_empty_exported)
        and bool(len(non_empty_exported) == len(exported_original_indices)),
        "drop_incomplete_events": bool(drop_incomplete_events),
        "concat_hypnogram_for_event": bool(concat_hypnogram_for_event),
        "window_size_hours": window_size_hours,
        "step_size_hours": step_size_hours,
        "rem_stage": rem_stage,
        "sample_output_csv": output_csv_path,
        "nanpad_output_csv": output_csv_nanpad_path,
        "metadata_csv": metadata_csv_path,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Export complete ===")
    print(f"Events requested: {len(rows)}")
    print(f"Events exported: {len(exported_original_indices)}")
    print(f"Final missing (rat,date): {len(missing_pairs)}")
    print(f"Output matrix shape: ({len(exported_original_indices)}, {padded_vector_length})")
    print(f"Numeric CSV: {output_csv_path}")
    print(f"NaN padded CSV: {output_csv_nanpad_path}")
    print(f"Metadata CSV: {metadata_csv_path}")
    print(f"Summary JSON: {summary_json_path}")

    return {
        "summary": summary,
        "metadata": metadata_df,
        "X_zero": X_zero,
        "X_nan": X_nan,
        "rows": rows,
        "exported_original_indices": np.array(exported_original_indices, dtype=int),
        "true_vector_lengths": np.array(true_vector_lengths, dtype=int),
        "missing_pairs": missing_pairs,
        "paths": {
            "sample_output_csv": output_csv_path,
            "nanpad_output_csv": output_csv_nanpad_path,
            "metadata_csv": metadata_csv_path,
            "summary_json": summary_json_path,
        },
    }


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


def _parse_feature_selection(feature_selection) -> dict[str, list[str]]:
    base_metrics = {"range", "mean", "std", "skewness", "kurtosis"}
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
        return out

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
        return out

    raise ValueError(
        "feature_selection must be dict or list/tuple/set of strings, e.g. "
        "{'even':['mean','range'], 'odd':['kurtosis','skewness']} or "
        "['mean even','range even','kurtosis odd','skewness odd']."
    )


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _default_parameter_selection() -> dict[str, dict]:
    """Per-metric defaults; optional ``g_prior`` scales Normal priors on regime means (Zellner-style).

    ``g_prior`` keys:
    - ``type``: ``none`` | ``unit_information`` | ``hyper_g_n`` | ``zellner_siow``
    - ``n``: effective sample size (default: number of rows in observed matrix for that block)
    - ``a``, ``b``: Beta hyperparameters for ``hyper_g_n`` (default 3, 3 as in Liang et al. style demos)
    """
    g_none = {"type": "none"}
    return {
        "mean": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 1.5},
            "sigma_prior": {"dist": "halfnormal", "sigma": 1.0},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
            "g_prior": dict(g_none),
        },
        "range": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -0.3, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.7},
            "g_prior": dict(g_none),
        },
        "std": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -0.7, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.7},
            "g_prior": dict(g_none),
        },
        "skewness": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 2.5},
            "sigma_prior": {"dist": "halfstudentt", "nu": 4.0, "sigma": 1.5},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
            "g_prior": dict(g_none),
        },
        "kurtosis": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 3.0},
            "sigma_prior": {"dist": "halfstudentt", "nu": 4.0, "sigma": 2.0},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
            "g_prior": dict(g_none),
        },
    }


# Named feature-selection layouts (multi-group mixes share one tau; chunk counts must match).
FEATURE_SELECTION_PRESETS: dict[str, dict[str, list[str]]] = {
    "concat_mean_odd_range": {"concat": ["mean"], "odd": ["range"]},
    "concat_range_odd_mean": {"concat": ["range"], "odd": ["mean"]},
    "concat_mean_even_range": {"concat": ["mean"], "even": ["range"]},
    "odd_mean_even_range": {"odd": ["mean"], "even": ["range"]},
    "concat_mean_range": {"concat": ["mean", "range"]},
    "odd_even_mean_range": {"odd": ["mean", "range"], "even": ["mean", "range"]},
}


def parameter_selection_with_g_prior(
    base: dict[str, dict] | None,
    g_type: str,
    *,
    n: int | None = None,
    a: float = 3.0,
    b: float = 3.0,
) -> dict[str, dict]:
    """Return a copy of ``base`` (or defaults) with ``g_prior`` set for every metric key present."""
    defaults = _default_parameter_selection()
    src = dict(base) if base else {}
    out: dict[str, dict] = {}
    for feat_name in defaults:
        merged = _deep_merge_dict(defaults[feat_name], src.get(feat_name, {}) or {})
        gp = dict(merged.get("g_prior") or {})
        gp["type"] = str(g_type).strip().lower()
        if n is not None:
            gp["n"] = int(n)
        gp.setdefault("a", float(a))
        gp.setdefault("b", float(b))
        merged["g_prior"] = gp
        out[feat_name] = merged
    return out


def _parse_parameter_selection(parameter_selection, active_features: set[str]) -> dict[str, dict]:
    """Return fully expanded per-feature distribution config."""
    defaults = _default_parameter_selection()
    if parameter_selection is None:
        parameter_selection = {}
    if not isinstance(parameter_selection, dict):
        raise ValueError("parameter_selection must be dict keyed by feature name.")

    unknown = sorted(set(parameter_selection.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            f"Unknown feature keys in parameter_selection: {unknown}. "
            f"Use one of: {sorted(defaults.keys())}"
        )

    out: dict[str, dict] = {}
    for feature_name in sorted(active_features):
        base_cfg = defaults[feature_name]
        custom_cfg = parameter_selection.get(feature_name, {})
        if custom_cfg is None:
            custom_cfg = {}
        if not isinstance(custom_cfg, dict):
            raise ValueError(f"parameter_selection['{feature_name}'] must be a dict.")
        out[feature_name] = _deep_merge_dict(base_cfg, custom_cfg)
    return out


def _build_prior(var_name: str, spec: dict, *, positive_only: bool = False):
    if not isinstance(spec, dict):
        raise ValueError(f"Prior spec for '{var_name}' must be a dict.")
    dist = str(spec.get("dist", "")).strip().lower()
    if not dist:
        raise ValueError(f"Prior spec for '{var_name}' must include 'dist'.")

    positive_dists = {"halfnormal", "halfstudentt", "exponential", "lognormal", "exponential_plus"}
    if positive_only and dist not in positive_dists:
        raise ValueError(
            f"Prior '{var_name}' must be positive; use one of {sorted(positive_dists)}, got '{dist}'."
        )

    if dist == "normal":
        return pm.Normal(var_name, mu=float(spec.get("mu", 0.0)), sigma=float(spec.get("sigma", 1.0)))
    if dist == "halfnormal":
        return pm.HalfNormal(var_name, sigma=float(spec.get("sigma", 1.0)))
    if dist == "halfstudentt":
        return pm.HalfStudentT(
            var_name,
            nu=float(spec.get("nu", 4.0)),
            sigma=float(spec.get("sigma", 1.0)),
        )
    if dist == "exponential":
        return pm.Exponential(var_name, lam=float(spec.get("lam", 1.0)))
    if dist == "lognormal":
        return pm.LogNormal(
            var_name,
            mu=float(spec.get("mu", 0.0)),
            sigma=float(spec.get("sigma", 1.0)),
        )
    if dist == "exponential_plus":
        lam = float(spec.get("lam", 0.05))
        offset = float(spec.get("offset", 2.0))
        raw = pm.Exponential(f"{var_name}_raw", lam=lam)
        return pm.Deterministic(var_name, raw + offset)

    raise ValueError(
        f"Unsupported prior dist '{dist}' for '{var_name}'. "
        "Use one of: normal, halfnormal, halfstudentt, exponential, lognormal, exponential_plus."
    )


def _g_multiplier(name_prefix: str, n_obs_rows: int, g_prior: dict | None):
    """Scalar g for Zellner-style scaling of Normal mu priors (Chapter 8 style flexible g).

    - ``none``: g = 1 (standard fixed prior spread).
    - ``unit_information``: g = n (fixed).
    - ``zellner_siow``: n/g ~ Gamma(1/2, 1/2)  =>  g = n / (n/g).
    - ``hyper_g_n``: u = 1/(1+n/g) ~ Beta(a/2, b/2)  =>  g = n*u/(1-u).

    Prior std on each regime mean uses ``mu_prior['sigma'] * sqrt(g)``.
    """
    if not g_prior:
        g_prior = {}
    typ = str(g_prior.get("type", "none")).strip().lower()
    if typ in {"", "none", "off", "fixed"}:
        return pt.as_tensor_variable(np.asarray(1.0, dtype=np.float64))

    n = int(g_prior.get("n", n_obs_rows))
    n = max(n, 1)
    n_f = float(n)

    if typ == "unit_information":
        return pt.as_tensor_variable(np.asarray(n_f, dtype=np.float64))

    if typ == "zellner_siow":
        n_over_g = pm.Gamma(f"{name_prefix}_n_over_g", alpha=0.5, beta=0.5)
        return pm.Deterministic(f"{name_prefix}_g", n_f / (n_over_g + 1e-12))

    if typ == "hyper_g_n":
        a = float(g_prior.get("a", 3.0))
        b = float(g_prior.get("b", 3.0))
        u = pm.Beta(f"{name_prefix}_u_hyper_gn", alpha=a * 0.5, beta=b * 0.5)
        one_m_u = pt.clip(1.0 - u, 1e-6, 1.0)
        return pm.Deterministic(f"{name_prefix}_g", n_f * u / one_m_u)

    raise ValueError(
        f"Unknown g_prior type '{typ}' for '{name_prefix}'. "
        "Use one of: none, unit_information, hyper_g_n, zellner_siow."
    )


def _build_mu_regime_normals(
    group_name: str,
    feat_name: str,
    spec: dict,
    *,
    n_obs_rows: int,
) -> tuple[object, object]:
    """Normal priors on mu_1, mu_2 for location parameter (Normal / StudentT / LogNormal mu)."""
    mu_spec = spec.get("mu_prior", {"dist": "normal", "mu": 0.0, "sigma": 2.0})
    if not isinstance(mu_spec, dict):
        raise ValueError(f"mu_prior for {group_name}/{feat_name} must be a dict.")
    b0_1 = float(mu_spec.get("mu", 0.0))
    b0_2 = float(mu_spec.get("mu_2", mu_spec.get("mu", 0.0)))
    s0 = float(mu_spec.get("sigma", 1.0))
    if s0 <= 0.0:
        raise ValueError(f"mu_prior.sigma must be > 0 for {group_name}/{feat_name}.")

    prefix = f"g_{group_name}_{feat_name}"
    g_prior = spec.get("g_prior")
    g = _g_multiplier(prefix, n_obs_rows, g_prior if isinstance(g_prior, dict) else None)
    scale = s0 * pt.sqrt(g)

    mu_1 = pm.Normal(f"mu_{group_name}_{feat_name}_1", mu=b0_1, sigma=scale)
    mu_2 = pm.Normal(f"mu_{group_name}_{feat_name}_2", mu=b0_2, sigma=scale)
    return mu_1, mu_2


def build_group_data(
    data_norm: np.ndarray,
    *,
    n_chunks: int,
    feature_selection,
) -> dict:
    """Build group_data dict used by changepoint model from feature_selection only."""

    feature_map = compute_chunk_feature_map(data_norm, n_chunks=n_chunks)
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
        "concat": concat_feature_map,
    }

    def _validate_same_chunk_count(out_dict: dict) -> None:
        counts = set()
        for feats in out_dict.values():
            for df in feats.values():
                counts.add(df.shape[1])
        if len(counts) > 1:
            raise ValueError(
                "Selected groups produce different chunk counts, incompatible with shared tau. "
                f"Chunk counts found: {sorted(counts)}."
            )

    selection = _parse_feature_selection(feature_selection)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for group_key, metric_list in selection.items():
        if group_key not in idx_map:
            raise ValueError(f"Unknown group '{group_key}' after parsing")
        out[group_key] = {}
        src = feature_source_map[group_key]
        for metric_name in metric_list:
            prefix = metric_name if group_key == "all" else f"{metric_name}_{group_key}"
            out[group_key][metric_name] = _to_feature_df(
                src[metric_name][:, idx_map[group_key]],
                prefix,
            )

    _validate_same_chunk_count(out)
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
    _RUNTIME_LAST_PREPARE_CFG = {
        "csv_path": csv_path,
        "bad_sample_indices": sorted(bad_set),
    }
    return {
        "data_raw": data_raw_model,
        "data_norm": data_norm_model,
        "data_raw_plot": data_raw_plot,
        "data_norm_plot": data_norm_plot,
        "good_indices": good_indices,
        "bad_set": bad_set,
        "n_samples_original": n_samples,
    }


_RUNTIME_DATA_NORM: np.ndarray | None = None
_RUNTIME_LAST_EXPORT_CFG: dict | None = None
_RUNTIME_LAST_PREPARE_CFG: dict | None = None



def set_runtime_data_norm(data_norm: np.ndarray) -> None:
    """Set default data_norm for run_variant() when not passed explicitly."""
    global _RUNTIME_DATA_NORM
    _RUNTIME_DATA_NORM = data_norm


def build_changepoint_model(
    group_data: dict,
    tau_lower: int = 2,
    tau_upper: int | None = None,
    parameter_selection: dict | None = None,
    tau_mode: str = "discrete",
):
    """Build PyMC changepoint model for group_data features.

    Per-metric ``parameter_selection[feat]['g_prior']`` (optional, default ``type: none``)
    scales the **Normal** prior standard deviation on regime means ``mu_*`` by ``sqrt(g)``,
    where ``g`` follows a unit-information, Zellner–Siow, or hyper-``g``/``n`` construction
    (flexible g-prior family from Bayesian model choice literature). Applies to
    ``normal`` / ``student_t`` / ``lognormal`` likelihood blocks (location ``mu``).
    """
    first_group = next(iter(group_data.values()))
    first_feat = next(iter(first_group.values()))
    n_group_chunks = first_feat.shape[1]

    if tau_upper is None:
        tau_upper = n_group_chunks
    if not (1 <= tau_lower <= tau_upper - 1 <= n_group_chunks):
        raise ValueError(
            f"Некорректный диапазон tau: [{tau_lower}, {tau_upper}], допустимо [1, {n_group_chunks}]"
        )

    active_features = {feat for features in group_data.values() for feat in features.keys()}
    parameter_cfg = _parse_parameter_selection(parameter_selection, active_features)

    tau_mode = str(tau_mode).strip().lower()
    if tau_mode not in {"discrete", "marginalized"}:
        raise ValueError("Unsupported tau_mode. Use one of: 'discrete', 'marginalized'.")

    with pm.Model() as model:
        idx = np.arange(n_group_chunks)
        tau_values = np.arange(tau_lower, tau_upper + 1, dtype=np.int64)
        n_tau = tau_values.size

        if tau_mode == "discrete":
            tau = pm.DiscreteUniform("tau", lower=tau_lower, upper=tau_upper)
            loglik_by_tau = None
            mask_before = None
        else:
            # For each candidate tau value, mark chunks that belong to "before tau" regime.
            mask_before = (idx[None, :] < (tau_values[:, None] - 1)).astype(float)
            loglik_by_tau = np.zeros(n_tau, dtype=float)

        for group_name, features in group_data.items():
            for feat_name, observed_df in features.items():
                observed = observed_df.to_numpy()
                spec = parameter_cfg[feat_name]
                likelihood = str(spec.get("likelihood", "normal")).strip().lower()

                if likelihood in {"normal", "student_t", "lognormal"}:
                    n_obs_rows = int(observed.shape[0])
                    mu_1, mu_2 = _build_mu_regime_normals(
                        group_name,
                        feat_name,
                        spec,
                        n_obs_rows=n_obs_rows,
                    )
                    sigma_1 = _build_prior(
                        f"sigma_{group_name}_{feat_name}_1",
                        spec.get("sigma_prior", {"dist": "halfnormal", "sigma": 1.0}),
                        positive_only=True,
                    )
                    sigma_2 = _build_prior(
                        f"sigma_{group_name}_{feat_name}_2",
                        spec.get("sigma_prior", {"dist": "halfnormal", "sigma": 1.0}),
                        positive_only=True,
                    )

                    if likelihood == "normal":
                        if tau_mode == "discrete":
                            mu = pm.math.switch(tau > idx + 1, mu_1, mu_2)
                            sigma = pm.math.switch(tau > idx + 1, sigma_1, sigma_2)
                            pm.Normal(
                                f"obs_{group_name}_{feat_name}",
                                mu=mu,
                                sigma=sigma,
                                observed=observed,
                            )
                        else:
                            ll_1 = pm.logp(pm.Normal.dist(mu=mu_1, sigma=sigma_1), observed)
                            ll_2 = pm.logp(pm.Normal.dist(mu=mu_2, sigma=sigma_2), observed)
                            ll_1_chunk = ll_1.sum(axis=0)
                            ll_2_chunk = ll_2.sum(axis=0)
                            ll_tau = (mask_before * ll_1_chunk[None, :] + (1.0 - mask_before) * ll_2_chunk[None, :]).sum(axis=1)
                            loglik_by_tau = loglik_by_tau + ll_tau
                    elif likelihood == "student_t":
                        nu = _build_prior(
                            f"nu_{group_name}_{feat_name}",
                            spec.get("nu_prior", {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0}),
                            positive_only=True,
                        )
                        if tau_mode == "discrete":
                            mu = pm.math.switch(tau > idx + 1, mu_1, mu_2)
                            sigma = pm.math.switch(tau > idx + 1, sigma_1, sigma_2)
                            pm.StudentT(
                                f"obs_{group_name}_{feat_name}",
                                nu=nu,
                                mu=mu,
                                sigma=sigma,
                                observed=observed,
                            )
                        else:
                            ll_1 = pm.logp(pm.StudentT.dist(nu=nu, mu=mu_1, sigma=sigma_1), observed)
                            ll_2 = pm.logp(pm.StudentT.dist(nu=nu, mu=mu_2, sigma=sigma_2), observed)
                            ll_1_chunk = ll_1.sum(axis=0)
                            ll_2_chunk = ll_2.sum(axis=0)
                            ll_tau = (mask_before * ll_1_chunk[None, :] + (1.0 - mask_before) * ll_2_chunk[None, :]).sum(axis=1)
                            loglik_by_tau = loglik_by_tau + ll_tau
                    else:
                        if tau_mode == "discrete":
                            mu = pm.math.switch(tau > idx + 1, mu_1, mu_2)
                            sigma = pm.math.switch(tau > idx + 1, sigma_1, sigma_2)
                            pm.LogNormal(
                                f"obs_{group_name}_{feat_name}",
                                mu=mu,
                                sigma=sigma,
                                observed=observed,
                            )
                        else:
                            ll_1 = pm.logp(pm.LogNormal.dist(mu=mu_1, sigma=sigma_1), observed)
                            ll_2 = pm.logp(pm.LogNormal.dist(mu=mu_2, sigma=sigma_2), observed)
                            ll_1_chunk = ll_1.sum(axis=0)
                            ll_2_chunk = ll_2.sum(axis=0)
                            ll_tau = (mask_before * ll_1_chunk[None, :] + (1.0 - mask_before) * ll_2_chunk[None, :]).sum(axis=1)
                            loglik_by_tau = loglik_by_tau + ll_tau
                    continue

                if likelihood == "gamma":
                    alpha_1 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_1",
                        spec.get("alpha_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    alpha_2 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_2",
                        spec.get("alpha_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    beta_1 = _build_prior(
                        f"beta_{group_name}_{feat_name}_1",
                        spec.get("beta_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    beta_2 = _build_prior(
                        f"beta_{group_name}_{feat_name}_2",
                        spec.get("beta_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    if tau_mode == "discrete":
                        alpha = pm.math.switch(tau > idx + 1, alpha_1, alpha_2)
                        beta = pm.math.switch(tau > idx + 1, beta_1, beta_2)
                        pm.Gamma(
                            f"obs_{group_name}_{feat_name}",
                            alpha=alpha,
                            beta=beta,
                            observed=observed,
                        )
                    else:
                        ll_1 = pm.logp(pm.Gamma.dist(alpha=alpha_1, beta=beta_1), observed)
                        ll_2 = pm.logp(pm.Gamma.dist(alpha=alpha_2, beta=beta_2), observed)
                        ll_1_chunk = ll_1.sum(axis=0)
                        ll_2_chunk = ll_2.sum(axis=0)
                        ll_tau = (mask_before * ll_1_chunk[None, :] + (1.0 - mask_before) * ll_2_chunk[None, :]).sum(axis=1)
                        loglik_by_tau = loglik_by_tau + ll_tau
                    continue

                if likelihood == "beta":
                    alpha_1 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_1",
                        spec.get("alpha_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    alpha_2 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_2",
                        spec.get("alpha_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    beta_1 = _build_prior(
                        f"beta_{group_name}_{feat_name}_1",
                        spec.get("beta_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    beta_2 = _build_prior(
                        f"beta_{group_name}_{feat_name}_2",
                        spec.get("beta_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    # Beta support is (0, 1), so clip normalized data slightly
                    # away from boundaries to avoid -inf logp at exact 0/1.
                    beta_eps = float(spec.get("eps", 1e-6))
                    observed_beta = np.clip(observed, beta_eps, 1.0 - beta_eps)
                    if tau_mode == "discrete":
                        alpha = pm.math.switch(tau > idx + 1, alpha_1, alpha_2)
                        beta = pm.math.switch(tau > idx + 1, beta_1, beta_2)
                        pm.Beta(
                            f"obs_{group_name}_{feat_name}",
                            alpha=alpha,
                            beta=beta,
                            observed=observed_beta,
                        )
                    else:
                        ll_1 = pm.logp(pm.Beta.dist(alpha=alpha_1, beta=beta_1), observed_beta)
                        ll_2 = pm.logp(pm.Beta.dist(alpha=alpha_2, beta=beta_2), observed_beta)
                        ll_1_chunk = ll_1.sum(axis=0)
                        ll_2_chunk = ll_2.sum(axis=0)
                        ll_tau = (mask_before * ll_1_chunk[None, :] + (1.0 - mask_before) * ll_2_chunk[None, :]).sum(axis=1)
                        loglik_by_tau = loglik_by_tau + ll_tau
                    continue

                raise ValueError(
                    f"Unsupported likelihood '{likelihood}' for feature '{feat_name}'. "
                    "Use one of: normal, student_t, lognormal, gamma, beta."
                )

        if tau_mode == "marginalized":
            # p(y|theta) = logsumexp_k [ log p(y|tau=k, theta) + log p(tau=k) ]
            log_w = loglik_by_tau - np.log(float(n_tau))
            # Scalar joint log-likelihood (one "observation" for WAIC) — tau is marginalized.
            pm.Deterministic("changepoint_joint_log_lik", pm.math.logsumexp(log_w))
            pm.Potential("tau_marginalized_logp", pm.math.logsumexp(log_w))
            tau_probs = pm.Deterministic("tau_probs", pm.math.softmax(log_w))
            tau_support = pm.Deterministic(
                "tau_support",
                pt.as_tensor_variable(tau_values.astype(np.float64)),
            )
            pm.Deterministic("tau_mean", pm.math.sum(tau_probs * tau_support))

    return model


def sample_model(
    model,
    draws: int = 4000,
    tune: int = 2000,
    *,
    nuts_backend: str = "pymc",
    chains: int = 4,
    cores: int | None = None,
):
    """Run MCMC sampling and return MultiTrace.

    Parameters
    ----------
    nuts_backend:
        - "pymc" (default): classic PyMC NUTS
        - "numpyro": JAX/NumPyro NUTS backend (can use GPU)
        - "blackjax": JAX/BlackJAX NUTS backend (can use GPU)
    """
    backend = str(nuts_backend).lower().strip()
    sample_kwargs = dict(
        draws=draws,
        tune=tune,
        return_inferencedata=False,
        compute_convergence_checks=False,
        target_accept=0.9,
        chains=chains,
    )
    if cores is not None:
        sample_kwargs["cores"] = cores

    with model:
        if backend == "pymc":
            trace = pm.sample(**sample_kwargs)
        elif backend in {"numpyro", "blackjax"}:
            import jax

            # With a single device, JAX "parallel" chains via pmap may fail for chains>1.
            device_count = int(jax.device_count())
            nuts_sampler_kwargs: dict[str, object] = {}
            if chains > 1 and device_count < chains:
                nuts_sampler_kwargs["chain_method"] = "vectorized"
            # BlackJAX progress bar uses IO callbacks, which can fail under vectorized/vmap.
            sample_kwargs["progressbar"] = False

            try:
                trace = pm.sample(
                    nuts_sampler=backend,
                    nuts_sampler_kwargs=nuts_sampler_kwargs,
                    **sample_kwargs,
                )
            except ValueError as exc:
                if "Model can not be sampled with NUTS alone" in str(exc):
                    raise ValueError(
                        "JAX backend requires a fully continuous differentiable model. "
                        "Use tau_mode='marginalized' (or backend='pymc')."
                    ) from exc
                raise
        else:
            raise ValueError(
                "Unsupported nuts_backend. Use one of: 'pymc', 'numpyro', 'blackjax'."
            )
    return trace


def _is_inferencedata(trace) -> bool:
    return isinstance(trace, az.InferenceData)


def _available_varnames(trace) -> set[str]:
    if _is_inferencedata(trace):
        return set(trace.posterior.data_vars)
    return set(trace.varnames)


def _values_flat(trace, var_name: str) -> np.ndarray:
    if _is_inferencedata(trace):
        return np.asarray(trace.posterior[var_name]).reshape(-1)
    return np.asarray(trace[var_name]).reshape(-1)


def _values_by_chain(trace, var_name: str) -> list[np.ndarray]:
    if _is_inferencedata(trace):
        arr = np.asarray(trace.posterior[var_name])
        return [arr[i] for i in range(arr.shape[0])]
    return trace.get_values(var_name, combine=False)


def _sampler_stat(trace, stat_name: str):
    if _is_inferencedata(trace):
        if hasattr(trace, "sample_stats") and stat_name in trace.sample_stats:
            return np.asarray(trace.sample_stats[stat_name])
        raise KeyError(f"Sampler stat '{stat_name}' not found in InferenceData.sample_stats")
    return trace.get_sampler_stats(stat_name, combine=True)


def summary_from_trace(trace, var_names):
    """Build ArviZ summary (mean, sd, r_hat, ESS)."""
    if _is_inferencedata(trace):
        return az.summary(trace, var_names=var_names)

    posterior = {}
    for var in var_names:
        chains = _values_by_chain(trace, var)
        posterior[var] = np.stack(chains, axis=0)
    idata = az.from_dict(posterior=posterior)
    return az.summary(idata, var_names=var_names)


def tau_probabilities(trace):
    """Return tau support and probabilities P(tau=k)."""
    trace_vars = _available_varnames(trace)
    if "tau" in trace_vars:
        tau_values = _values_flat(trace, "tau").astype(int).ravel()
        tau_min, tau_max = int(tau_values.min()), int(tau_values.max())
        support = np.arange(tau_min, tau_max + 1)
        counts = np.bincount(tau_values, minlength=tau_max + 1)[tau_min : tau_max + 1]
        probs = counts / counts.sum()
        return support, probs

    if "tau_probs" in trace_vars and "tau_support" in trace_vars:
        if _is_inferencedata(trace):
            probs_draws = np.asarray(trace.posterior["tau_probs"], dtype=float)
            support_draws = np.asarray(trace.posterior["tau_support"], dtype=float)
            probs = probs_draws.mean(axis=(0, 1))
            support = support_draws[0, 0].astype(int).ravel()
        else:
            probs_draws = np.asarray(trace["tau_probs"], dtype=float)
            support_draws = np.asarray(trace["tau_support"], dtype=float)
            probs = probs_draws.mean(axis=0)
            support = support_draws[0].astype(int).ravel() if support_draws.ndim > 1 else support_draws.astype(int).ravel()
        probs = probs / probs.sum()
        return support, probs

    raise ValueError("Neither 'tau' nor ('tau_probs' and 'tau_support') found in trace.")


def _positive_feature_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    return arr


def _profile_x_grid(observed: np.ndarray, likelihood: str, grid_size: int) -> np.ndarray:
    finite = np.asarray(observed, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Observed feature array has no finite values.")

    if likelihood in {"lognormal", "gamma"}:
        finite = finite[finite > 0.0]
        if finite.size == 0:
            raise ValueError(
                "Observed feature array has no positive values required by positive-only likelihood."
            )
        x_min = max(float(np.min(finite)) * 0.8, 1e-6)
        x_max = float(np.max(finite)) * 1.2
        return np.linspace(x_min, x_max, grid_size)

    if likelihood == "beta":
        # Beta support is strictly (0, 1); use a fixed in-support grid.
        return np.linspace(1e-6, 1.0 - 1e-6, grid_size)

    q1 = float(np.quantile(finite, 0.01))
    q99 = float(np.quantile(finite, 0.99))
    spread = q99 - q1
    if spread <= 0.0:
        spread = max(float(np.std(finite)), 1e-3)
    x_min = q1 - 0.25 * spread
    x_max = q99 + 0.25 * spread
    return np.linspace(x_min, x_max, grid_size)


def _likelihood_pdf_from_posterior(
    *,
    likelihood: str,
    x: np.ndarray,
    params_1: dict,
    params_2: dict,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    eps = 1e-12
    likelihood = str(likelihood).strip().lower()

    if likelihood == "normal":
        mu_1 = float(params_1["mu"])
        mu_2 = float(params_2["mu"])
        sigma_1 = max(float(params_1["sigma"]), eps)
        sigma_2 = max(float(params_2["sigma"]), eps)
        c1 = 1.0 / (sigma_1 * np.sqrt(2.0 * np.pi))
        c2 = 1.0 / (sigma_2 * np.sqrt(2.0 * np.pi))
        y1 = c1 * np.exp(-0.5 * ((x - mu_1) / sigma_1) ** 2)
        y2 = c2 * np.exp(-0.5 * ((x - mu_2) / sigma_2) ** 2)
        return y1, y2

    if likelihood == "student_t":
        nu = max(float(params_1["nu"]), 2.0 + eps)
        mu_1 = float(params_1["mu"])
        mu_2 = float(params_2["mu"])
        sigma_1 = max(float(params_1["sigma"]), eps)
        sigma_2 = max(float(params_2["sigma"]), eps)

        # Student-t PDF via log-space for stability:
        # log f(x) = lgamma((nu+1)/2)-lgamma(nu/2)-0.5*log(nu*pi)-log(sigma)
        #           -((nu+1)/2)*log(1 + ((x-mu)^2)/(nu*sigma^2))
        def _student_t_pdf(xv: np.ndarray, mu: float, sigma: float) -> np.ndarray:
            log_c = (
                float(math.lgamma((nu + 1.0) / 2.0))
                - float(math.lgamma(nu / 2.0))
                - 0.5 * np.log(nu * np.pi)
                - np.log(sigma)
            )
            z2 = ((xv - mu) / sigma) ** 2
            return np.exp(log_c - ((nu + 1.0) / 2.0) * np.log1p(z2 / nu))

        return _student_t_pdf(x, mu_1, sigma_1), _student_t_pdf(x, mu_2, sigma_2)

    if likelihood == "lognormal":
        mu_1 = float(params_1["mu"])
        mu_2 = float(params_2["mu"])
        sigma_1 = max(float(params_1["sigma"]), eps)
        sigma_2 = max(float(params_2["sigma"]), eps)
        x_pos = np.maximum(x, eps)
        c1 = 1.0 / (x_pos * sigma_1 * np.sqrt(2.0 * np.pi))
        c2 = 1.0 / (x_pos * sigma_2 * np.sqrt(2.0 * np.pi))
        y1 = c1 * np.exp(-((np.log(x_pos) - mu_1) ** 2) / (2.0 * sigma_1**2))
        y2 = c2 * np.exp(-((np.log(x_pos) - mu_2) ** 2) / (2.0 * sigma_2**2))
        return y1, y2

    if likelihood == "gamma":
        alpha_1 = max(float(params_1["alpha"]), eps)
        alpha_2 = max(float(params_2["alpha"]), eps)
        beta_1 = max(float(params_1["beta"]), eps)
        beta_2 = max(float(params_2["beta"]), eps)
        x_pos = np.maximum(x, eps)
        y1 = (
            (beta_1**alpha_1)
            / np.exp(float(math.lgamma(alpha_1)))
            * x_pos ** (alpha_1 - 1.0)
            * np.exp(-beta_1 * x_pos)
        )
        y2 = (
            (beta_2**alpha_2)
            / np.exp(float(math.lgamma(alpha_2)))
            * x_pos ** (alpha_2 - 1.0)
            * np.exp(-beta_2 * x_pos)
        )
        return y1, y2

    if likelihood == "beta":
        alpha_1 = max(float(params_1["alpha"]), eps)
        alpha_2 = max(float(params_2["alpha"]), eps)
        beta_1 = max(float(params_1["beta"]), eps)
        beta_2 = max(float(params_2["beta"]), eps)
        x_unit = np.clip(x, eps, 1.0 - eps)
        y1 = (
            np.exp(float(math.lgamma(alpha_1 + beta_1)) - float(math.lgamma(alpha_1)) - float(math.lgamma(beta_1)))
            * (x_unit ** (alpha_1 - 1.0))
            * ((1.0 - x_unit) ** (beta_1 - 1.0))
        )
        y2 = (
            np.exp(float(math.lgamma(alpha_2 + beta_2)) - float(math.lgamma(alpha_2)) - float(math.lgamma(beta_2)))
            * (x_unit ** (alpha_2 - 1.0))
            * ((1.0 - x_unit) ** (beta_2 - 1.0))
        )
        return y1, y2

    raise ValueError(
        f"Unsupported likelihood '{likelihood}'. "
        "Use one of: normal, student_t, lognormal, gamma, beta."
    )


def feature_likelihood_profiles(
    trace,
    group_data: dict,
    parameter_selection: dict | None = None,
    *,
    grid_size: int = 300,
    plot: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return and optionally plot before/after likelihood profiles for each selected feature.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        Nested dict ``profiles[group_name][feat_name]`` with columns:
        ``x``, ``pdf_before``, ``pdf_after``.
    """
    if grid_size < 50:
        raise ValueError("grid_size must be >= 50.")

    trace_vars = _available_varnames(trace)
    active_features = {feat for features in group_data.values() for feat in features.keys()}
    parameter_cfg = _parse_parameter_selection(parameter_selection, active_features)
    profiles: dict[str, dict[str, pd.DataFrame]] = {}

    for group_name, features in group_data.items():
        profiles[group_name] = {}
        for feat_name, observed_df in features.items():
            likelihood = str(parameter_cfg[feat_name].get("likelihood", "normal")).strip().lower()
            observed = observed_df.to_numpy(dtype=float).reshape(-1)

            x = _profile_x_grid(observed, likelihood=likelihood, grid_size=grid_size)
            params_1: dict[str, float] = {}
            params_2: dict[str, float] = {}

            for p in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{p}_{group_name}_{feat_name}_1"
                p2 = f"{p}_{group_name}_{feat_name}_2"
                if p1 in trace_vars and p2 in trace_vars:
                    params_1[p] = float(np.mean(_values_flat(trace, p1)))
                    params_2[p] = float(np.mean(_values_flat(trace, p2)))

            nu_name = f"nu_{group_name}_{feat_name}"
            if nu_name in trace_vars:
                nu_mean = float(np.mean(_values_flat(trace, nu_name)))
                params_1["nu"] = nu_mean
                params_2["nu"] = nu_mean

            y_before, y_after = _likelihood_pdf_from_posterior(
                likelihood=likelihood,
                x=x,
                params_1=params_1,
                params_2=params_2,
            )
            profile_df = pd.DataFrame(
                {
                    "x": x,
                    "pdf_before": y_before,
                    "pdf_after": y_after,
                }
            )
            profiles[group_name][feat_name] = profile_df

            if plot:
                plt.figure(figsize=(7, 4))
                if likelihood in {"lognormal", "gamma"}:
                    obs_plot = _positive_feature_values(observed)
                else:
                    obs_plot = np.asarray(observed, dtype=float)
                    obs_plot = obs_plot[np.isfinite(obs_plot)]
                if obs_plot.size > 0:
                    plt.hist(
                        obs_plot,
                        bins=30,
                        density=True,
                        alpha=0.25,
                        color="green",
                        label="observed",
                    )
                plt.plot(x, y_before, color="#A60628", linewidth=2.0, label="до tau")
                plt.plot(x, y_after, color="#7A68A6", linewidth=2.0, label="после tau")
                plt.title(f"Likelihood profile: {group_name}/{feat_name} ({likelihood})")
                plt.xlabel("value")
                plt.ylabel("density")
                plt.grid(alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.show()

    return profiles


def plot_trace_and_tau(trace, trace_vars, title_prefix: str):
    """Plot traces for selected vars and tau bar chart."""
    if _is_inferencedata(trace):
        az.plot_trace(trace, var_names=trace_vars, compact=False)
    else:
        posterior = {}
        for var in trace_vars:
            chains = _values_by_chain(trace, var)
            posterior[var] = np.stack(chains, axis=0)
        idata = az.from_dict(posterior=posterior)
        az.plot_trace(idata, var_names=trace_vars, compact=False)
    plt.suptitle(f"Trace plots: {title_prefix}", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_posteriors_like_script(trace, group_data: dict, title_prefix: str):
    """Posterior histograms for before/after tau in script-like style."""
    trace_vars = _available_varnames(trace)
    pairs = []
    for group_name, features in group_data.items():
        for feat_name in features:
            for param_name in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in trace_vars and p2 in trace_vars:
                    pairs.append(((p1, p2), f"{group_name} {feat_name} {param_name}"))

    n_axes = len(pairs) + 1
    n_cols = 3
    n_rows = (n_axes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    color_before, color_after = "#A60628", "#7A68A6"
    for i, ((name_1, name_2), title) in enumerate(pairs):
        ax = axes[i]
        # `edgecolor/linewidth` adds visible "borders" around histogram bins.
        ax.hist(
            _values_flat(trace, name_1),
            bins=30,
            alpha=0.6,
            density=True,
            color=color_before,
            edgecolor="black",
            linewidth=0.6,
            label="до tau",
        )
        ax.hist(
            _values_flat(trace, name_2),
            bins=30,
            alpha=0.6,
            density=True,
            color=color_after,
            edgecolor="black",
            linewidth=0.6,
            label="после tau",
        )
        ax.set_title(title)
        ax.legend()

    tau_idx = len(pairs)
    support, probs = tau_probabilities(trace)
    axes[tau_idx].bar(support, probs, color="#348ABD", width=0.8, edgecolor="black", linewidth=0.6)
    axes[tau_idx].set_xticks(support)
    axes[tau_idx].set_title("tau")
    axes[tau_idx].set_xlabel("индекс группы чанков")

    for j in range(tau_idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Постериоры (до/после tau): {title_prefix}", y=1.02)
    plt.tight_layout()
    plt.show()

    support, probs = tau_probabilities(trace)
    plt.figure(figsize=(6, 3.5))
    plt.bar(support, probs, width=0.8)
    plt.xticks(support)
    plt.xlabel("tau (индекс группы чанков)")
    plt.ylabel("P(tau=k)")
    plt.title(f"Постериорное распределение tau: {title_prefix}")
    plt.tight_layout()
    plt.show()


def changepoint_model_config_fingerprint(config: dict) -> str:
    """Stable hash for caching / deduplication of discrete model configurations."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _posterior_dict_from_trace(trace) -> dict[str, np.ndarray]:
    """Stack posterior samples as (chain, draw, *shape) for ArviZ."""
    names = sorted(_available_varnames(trace))
    posterior: dict[str, np.ndarray] = {}
    for name in names:
        try:
            chains = _values_by_chain(trace, name)
            posterior[name] = np.stack(chains, axis=0)
        except Exception:
            continue
    return posterior


def _float_ic_scalar(val: Any) -> float:
    try:
        if val is None:
            return float("nan")
        return float(np.asarray(val, dtype=float).squeeze())
    except Exception:
        return float("nan")


def _idata_for_waic_from_trace(trace, model) -> az.InferenceData:
    """Build InferenceData with a log_likelihood group suitable for ``az.waic`` / ``az.loo``."""
    posterior = _posterior_dict_from_trace(trace)
    if not posterior:
        raise ValueError("No posterior variables found in trace for WAIC/LOO.")

    if "changepoint_joint_log_lik" in _available_varnames(trace):
        ll_chains = _values_by_chain(trace, "changepoint_joint_log_lik")
        ll = np.stack(ll_chains, axis=0).astype(float)
        if ll.ndim == 2:
            ll = ll[..., np.newaxis]
        loglik_group = {"changepoint_joint_log_lik": ll}
        return az.from_dict(posterior=posterior, log_likelihood=loglik_group)

    idata = az.from_dict(posterior=posterior)
    try:
        with model:
            idata = pm.compute_log_likelihood(idata, model=model)
    except Exception as exc:
        raise RuntimeError(
            "WAIC/LOO requires pointwise log-likelihood; compute_log_likelihood failed. "
            "For marginalized tau models use changepoint_joint_log_lik in the trace."
        ) from exc
    if not hasattr(idata, "log_likelihood") or not idata.log_likelihood:
        raise RuntimeError("compute_log_likelihood did not populate idata.log_likelihood.")
    return idata


def score_changepoint_trace(
    trace,
    *,
    group_data: dict,
    parameter_selection: dict | None,
    tau_threshold: float = 7.0,
    summary_var_names: List[str] | None = None,
    model=None,
    criterion: str = "waic",
    warn_on_fallback: bool = True,
) -> dict[str, Any]:
    """Summarize a changepoint trace for model comparison and Metropolis-Hastings scoring.

    Returns keys including: p_tau_gt_threshold, map_tau, map_tau_prob, tau_entropy,
    tau_concentration, r_hat_max, ess_min_bulk, ess_min_tail, n_divergences, bfmi / bfmi_approx,
    and when ``model`` is provided: elpd, waic / loo, p_waic / p_loo, criterion metadata.
    """
    support, probs = tau_probabilities(trace)
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    p_gt = float(probs[support > float(tau_threshold)].sum())
    map_idx = int(np.argmax(probs))
    map_tau = int(support[map_idx])
    map_p = float(probs[map_idx])
    ent = float(-np.sum(probs * np.log(probs + 1e-20)))
    max_ent = float(np.log(max(len(probs), 1)))
    conc = float(1.0 - ent / max_ent) if max_ent > 1e-9 else 0.0

    r_hat_max = 1.0
    ess_min_bulk = float("inf")
    ess_min_tail = float("inf")
    if summary_var_names:
        try:
            summ = summary_from_trace(trace, summary_var_names)
            if "r_hat" in summ.columns:
                rh = summ["r_hat"].to_numpy(dtype=float)
                rh = rh[np.isfinite(rh)]
                if rh.size:
                    r_hat_max = float(np.nanmax(rh))
            if "ess_bulk" in summ.columns:
                eb = summ["ess_bulk"].to_numpy(dtype=float)
                eb = eb[np.isfinite(eb)]
                if eb.size:
                    ess_min_bulk = float(np.nanmin(eb))
            if "ess_tail" in summ.columns:
                et = summ["ess_tail"].to_numpy(dtype=float)
                et = et[np.isfinite(et)]
                if et.size:
                    ess_min_tail = float(np.nanmin(et))
        except Exception:
            pass

    n_div = 0
    try:
        diverging = _sampler_stat(trace, "diverging")
        n_div = int(np.asarray(diverging).sum())
    except Exception:
        pass

    bfmi = float("nan")
    try:
        bfmi_arr = np.asarray(az.bfmi(trace), dtype=float).reshape(-1)
        bfmi_arr = bfmi_arr[np.isfinite(bfmi_arr)]
        if bfmi_arr.size:
            bfmi = float(np.mean(bfmi_arr))
    except Exception:
        try:
            energy = np.asarray(_sampler_stat(trace, "energy"), dtype=float).reshape(-1)
            if energy.size > 1 and np.var(energy) > 0:
                bfmi = float(np.mean(np.diff(energy) ** 2) / np.var(energy))
        except Exception:
            pass

    n_feat_blocks = sum(len(v) for v in group_data.values())
    active_features = sorted({feat for feats in group_data.values() for feat in feats.keys()})
    likelihoods: dict[str, str] = {}
    if parameter_selection:
        for f in active_features:
            if f in parameter_selection and isinstance(parameter_selection[f], dict):
                likelihoods[f] = str(parameter_selection[f].get("likelihood", "")).lower()

    crit = str(criterion).strip().lower()
    elpd = float("nan")
    waic_stat = float("nan")
    p_waic = float("nan")
    loo_stat = float("nan")
    p_loo = float("nan")
    criterion_error: str | None = None
    ic_computed = False

    if model is not None and crit in {"waic", "loo"}:
        try:
            idata_ic = _idata_for_waic_from_trace(trace, model)
            if crit == "waic":
                ic = az.waic(idata_ic, scale="log")
                elpd = _float_ic_scalar(getattr(ic, "elpd_waic", float("nan")))
                p_waic = _float_ic_scalar(getattr(ic, "p_waic", float("nan")))
                waic_stat = _float_ic_scalar(getattr(ic, "waic", float("nan")))
                if not math.isfinite(waic_stat) and math.isfinite(elpd):
                    waic_stat = float(-2.0 * elpd)
            else:
                ic = az.loo(idata_ic, scale="log")
                elpd = _float_ic_scalar(getattr(ic, "elpd_loo", float("nan")))
                p_loo = _float_ic_scalar(getattr(ic, "p_loo", float("nan")))
                loo_stat = _float_ic_scalar(getattr(ic, "loo", float("nan")))
                if not math.isfinite(loo_stat) and math.isfinite(elpd):
                    loo_stat = float(-2.0 * elpd)
            ic_computed = True
        except Exception as exc:
            criterion_error = str(exc)
            elpd = float("-inf")
            ic_computed = True
    elif model is None and warn_on_fallback:
        warnings.warn(
            "score_changepoint_trace: ``model`` is None; skipping WAIC/LOO. "
            "Metropolis-Hastings will use the legacy tau-based log-target unless you pass ``model``.",
            UserWarning,
            stacklevel=2,
        )

    out: dict[str, Any] = {
        "p_tau_gt_threshold": p_gt,
        "map_tau": map_tau,
        "map_tau_prob": map_p,
        "tau_entropy": ent,
        "tau_concentration": conc,
        "r_hat_max": r_hat_max,
        "ess_min_bulk": ess_min_bulk,
        "ess_min_tail": ess_min_tail,
        "n_divergences": n_div,
        "bfmi": bfmi,
        "bfmi_approx": bfmi,
        "n_feature_blocks": n_feat_blocks,
        "active_features": active_features,
        "likelihoods": likelihoods,
        "elpd": elpd,
        "criterion": crit if model is not None else "none",
        "ic_computed": ic_computed,
        "waic": waic_stat,
        "p_waic": p_waic,
        "loo": loo_stat,
        "p_loo": p_loo,
        "criterion_error": criterion_error,
    }
    return out


def changepoint_log_target(
    score_parts: dict[str, Any],
    *,
    w_elpd: float = 1.0,
    r_hat_gate: float = 1.05,
    ess_threshold: float = 100.0,
    bfmi_threshold: float = 0.3,
    w_p_tau: float = 0.2,
    w_map: float = 0.05,
    w_conc: float = 0.05,
    w_complexity: float = 0.05,
    w_rhat_penalty: float = 40.0,
    w_bfmi_penalty: float = 25.0,
    w_ess_penalty: float = 0.15,
    r_hat_gate_legacy: float = 1.01,
    w_p_tau_legacy: float = 8.0,
    w_map_legacy: float = 2.0,
    w_conc_legacy: float = 1.0,
    w_ess_legacy: float = 0.002,
    w_complexity_legacy: float = 0.08,
) -> float:
    """Log-scale score for Metropolis-Hastings; higher is better.

    When ``score_parts['ic_computed']`` is true (WAIC/LOO ran), **elpd** dominates; small
    bonuses use :math:`P(\\tau > \\text{thr})`, MAP :math:`\\tau` mass, and concentration.
    Otherwise the legacy tau- and ESS-weighted score is used (backward compatible).
    """
    p = float(score_parts.get("p_tau_gt_threshold", 0.0))
    mp = float(score_parts.get("map_tau_prob", 0.0))
    conc = float(score_parts.get("tau_concentration", 0.0))
    rmax = float(score_parts.get("r_hat_max", 1.0))
    essb = float(score_parts.get("ess_min_bulk", float("inf")))
    esst = float(score_parts.get("ess_min_tail", float("inf")))
    ndiv = int(score_parts.get("n_divergences", 0))
    nblk = int(score_parts.get("n_feature_blocks", 1))
    bfmi = float(score_parts.get("bfmi", score_parts.get("bfmi_approx", float("nan"))))

    if ndiv > 0:
        return float("-inf")

    ic_computed = bool(score_parts.get("ic_computed", False))
    logp = math.log(max(p, 1e-12))

    if ic_computed:
        elpd = float(score_parts.get("elpd", float("nan")))
        if not math.isfinite(elpd):
            return float("-inf")

        pen = 0.0
        if rmax > r_hat_gate:
            pen -= w_rhat_penalty * (rmax - r_hat_gate)

        ess_min = float("inf")
        if math.isfinite(essb):
            ess_min = min(ess_min, essb)
        if math.isfinite(esst):
            ess_min = min(ess_min, esst)
        if not math.isfinite(ess_min) or ess_min < ess_threshold:
            short = ess_threshold if not math.isfinite(ess_min) else max(0.0, ess_threshold - ess_min)
            pen -= w_ess_penalty * short

        if math.isfinite(bfmi) and bfmi < bfmi_threshold:
            pen -= w_bfmi_penalty * (bfmi_threshold - bfmi)

        return (
            w_elpd * elpd
            + w_p_tau * logp
            + w_map * math.log(max(mp, 1e-12))
            + w_conc * conc
            - w_complexity * float(max(nblk - 1, 0))
            + pen
        )

    gate = 0.0
    if rmax > r_hat_gate_legacy:
        gate -= 50.0 * (rmax - r_hat_gate_legacy)
    if not math.isfinite(essb) or essb < 100.0:
        gate -= 5.0
    ess_term = math.log(max(essb, 1.0)) if math.isfinite(essb) else 0.0

    return (
        w_p_tau_legacy * logp
        + w_map_legacy * math.log(max(mp, 1e-12))
        + w_conc_legacy * conc
        + w_ess_legacy * ess_term
        - w_complexity_legacy * float(max(nblk - 1, 0))
        + gate
    )


def _clone_config(config: dict) -> dict:
    return copy.deepcopy(config)


def _validate_feature_selection_for_n_chunks(feature_selection: dict, n_chunks: int) -> bool:
    try:
        wide = max(512, int(n_chunks) * 32)
        build_group_data(
            np.zeros((2, wide), dtype=float),
            n_chunks=n_chunks,
            feature_selection=feature_selection,
        )
        return True
    except Exception:
        return False


def propose_changepoint_model_config(
    current: dict,
    proposal_options: dict,
    rng: np.random.Generator,
) -> dict:
    """One symmetric random-walk proposal on the discrete model space.

    proposal_options keys:
    - rem_profile_choices: list[dict] with window_size_hours, step_size_hours, rem_stage
    - n_chunks_choices: list[int] (even if concat group may be used)
    - allowed_groups: list[str] subset of concat, odd, even, all
    - allowed_metrics: list[str]
    - likelihood_choices_by_metric: dict[metric, list[str]] optional; defaults used if missing

    Move types (uniform over 6): REM profile, ``n_chunks``, feature toggle, likelihood family,
    **swap** metric within a group (fixed model size), **perturb** a prior scale (symmetric
    multiplicative random walk on ``log sigma``).
    """
    prop = _clone_config(current)
    rem_choices = list(proposal_options.get("rem_profile_choices") or [])
    n_choices = list(proposal_options.get("n_chunks_choices") or [])
    groups = list(proposal_options.get("allowed_groups") or ["concat", "odd", "even"])
    metrics = list(proposal_options.get("allowed_metrics") or ["mean", "range"])
    like_map: dict[str, list[str]] = dict(proposal_options.get("likelihood_choices_by_metric") or {})
    default_likes = ["student_t", "normal", "lognormal", "gamma", "beta"]

    move = int(rng.integers(0, 6))
    n_chunks = int(prop["n_chunks"])

    if move == 0 and len(rem_choices) > 1:
        cur_rem = prop.get("rem_profile_params") or {}
        others = [r for r in rem_choices if r != cur_rem]
        if others:
            prop["rem_profile_params"] = dict(rng.choice(others))
        return prop

    if move == 1 and len(n_choices) > 1:
        others = [n for n in n_choices if int(n) != n_chunks]
        if others:
            prop["n_chunks"] = int(rng.choice(others))
        return prop

    if move == 2:
        fs = dict(prop.get("feature_selection") or {})
        pair_pool: List[Tuple[str, str]] = []
        for g in groups:
            for m in metrics:
                pair_pool.append((g, m))
        if not pair_pool:
            return prop
        g, m = pair_pool[int(rng.integers(0, len(pair_pool)))]
        if g not in fs:
            fs[g] = []
        lst = list(fs[g])
        if m in lst:
            lst = [x for x in lst if x != m]
            if lst:
                fs[g] = lst
            else:
                del fs[g]
        else:
            lst.append(m)
            fs[g] = sorted(set(lst), key=lambda x: metrics.index(x) if x in metrics else 0)
        if fs and _validate_feature_selection_for_n_chunks(fs, int(prop["n_chunks"])):
            prop["feature_selection"] = fs
        return prop

    if move == 3:
        fs = dict(prop.get("feature_selection") or {})
        ps = dict(prop.get("parameter_selection") or {})
        active: List[str] = []
        for feats in fs.values():
            for feat in feats:
                if feat not in active:
                    active.append(feat)
        if not active:
            return prop
        feat = str(rng.choice(active))
        cur_like = str(
            ps.get(feat, {}).get("likelihood", _default_parameter_selection()[feat].get("likelihood", "normal"))
        ).lower()
        if feat in like_map:
            choices = [str(c).lower() for c in like_map[feat]]
        else:
            choices = [c.lower() for c in default_likes]
        alts = [c for c in choices if c != cur_like]
        if not alts:
            return prop
        new_like = str(rng.choice(alts))
        if feat not in ps:
            ps[feat] = {}
        ps[feat] = dict(ps[feat])
        ps[feat]["likelihood"] = new_like
        prop["parameter_selection"] = ps
        return prop

    if move == 4:
        # Swap: replace one active metric in a group with a different allowed metric (same group).
        fs = dict(prop.get("feature_selection") or {})
        nonempty_groups = [g for g in groups if g in fs and fs[g]]
        if not nonempty_groups:
            return prop
        g = str(rng.choice(nonempty_groups))
        cur_metrics = list(fs[g])
        if not cur_metrics:
            return prop
        m_old = str(rng.choice(cur_metrics))
        alternatives = [m for m in metrics if m != m_old]
        if not alternatives:
            return prop
        m_new = str(rng.choice(alternatives))
        new_lst = [m for m in cur_metrics if m != m_old]
        if m_new not in new_lst:
            new_lst.append(m_new)
        fs[g] = sorted(set(new_lst), key=lambda x: metrics.index(x) if x in metrics else 0)
        if fs and _validate_feature_selection_for_n_chunks(fs, int(prop["n_chunks"])):
            prop["feature_selection"] = fs
        return prop

    if move == 5:
        # Perturb: symmetric random walk on log(scale) for ``mu_prior.sigma`` or ``sigma_prior.sigma``.
        fs = dict(prop.get("feature_selection") or {})
        ps = dict(prop.get("parameter_selection") or {})
        active = [f for feats in fs.values() for f in feats]
        active = list(dict.fromkeys(active))
        if not active:
            return prop
        feat = str(rng.choice(active))
        if feat not in ps:
            ps[feat] = {}
        ps[feat] = dict(ps[feat])
        key = str(rng.choice(["sigma_prior", "mu_prior"]))
        if key not in ps[feat]:
            ps[feat][key] = (
                {"dist": "halfnormal", "sigma": 1.0}
                if key == "sigma_prior"
                else {"dist": "normal", "mu": 0.0, "sigma": 1.0}
            )
        ps[feat][key] = dict(ps[feat][key])
        spin = float(np.exp(rng.uniform(-0.45, 0.45)))
        if key == "sigma_prior" and str(ps[feat][key].get("dist", "")).lower() in {"halfnormal", "halfstudentt"}:
            base = float(ps[feat][key].get("sigma", 1.0))
            ps[feat][key]["sigma"] = max(0.05, base * spin)
        elif key == "mu_prior" and str(ps[feat][key].get("dist", "")).lower() == "normal":
            base = float(ps[feat][key].get("sigma", 1.0))
            ps[feat][key]["sigma"] = max(0.05, base * spin)
        prop["parameter_selection"] = ps
        return prop

    return prop


def _build_summary_var_names(group_data: dict, trace) -> List[str]:
    available = _available_varnames(trace)
    out: List[str] = []
    if "tau" in available:
        out.append("tau")
    if "tau_mean" in available:
        out.append("tau_mean")
    for group_name, features in group_data.items():
        for feat_name in features:
            for param_name in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in available and p2 in available:
                    out.extend([p1, p2])
            nu_name = f"nu_{group_name}_{feat_name}"
            if nu_name in available:
                out.append(nu_name)
    return sorted(set(out))


def _fit_config_once(
    config: dict,
    *,
    data_norm: np.ndarray,
    draws: int,
    tune: int,
    nuts_backend: str,
    chains: int,
    cores: int | None,
    tau_mode: str,
    tau_lower: int,
    tau_upper: int | None,
    ic_criterion: str = "waic",
) -> Tuple[dict, Any, dict, Any]:
    """Build group_data, sample, return (group_data, trace, score_parts, model)."""
    group_data = build_group_data(
        data_norm,
        n_chunks=int(config["n_chunks"]),
        feature_selection=config["feature_selection"],
    )
    first_group = next(iter(group_data.values()))
    first_feat = next(iter(first_group.values()))
    n_group_chunks = first_feat.shape[1]
    tu = tau_upper if tau_upper is not None else n_group_chunks
    model = build_changepoint_model(
        group_data,
        tau_lower=tau_lower,
        tau_upper=tu,
        parameter_selection=config.get("parameter_selection"),
        tau_mode=tau_mode,
    )
    trace = sample_model(
        model,
        draws=draws,
        tune=tune,
        nuts_backend=nuts_backend,
        chains=chains,
        cores=cores,
    )
    summ_vars = _build_summary_var_names(group_data, trace)
    score_parts = score_changepoint_trace(
        trace,
        group_data=group_data,
        parameter_selection=config.get("parameter_selection"),
        tau_threshold=float(config.get("tau_threshold", 7.0)),
        summary_var_names=summ_vars if summ_vars else None,
        model=model,
        criterion=ic_criterion,
        warn_on_fallback=False,
    )
    return group_data, trace, score_parts, model


def _running_log_target_quantiles(
    chain_records: List[dict[str, Any]],
    window: int,
) -> tuple[float, float, float]:
    w = max(1, int(window))
    vals: List[float] = []
    for r in chain_records[-w:]:
        lt = r.get("log_target")
        if lt is None:
            continue
        v = float(lt)
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(np.quantile(a, 0.1)), float(np.quantile(a, 0.5)), float(np.quantile(a, 0.9))


def _best_fingerprint_by_elpd(score_cache: dict[str, dict[str, Any]]) -> tuple[str | None, float]:
    best_fp: str | None = None
    best_elpd = float("-inf")
    for fp, sc in score_cache.items():
        e = float(sc.get("elpd", float("-inf")))
        if math.isfinite(e) and e > best_elpd:
            best_elpd = e
            best_fp = str(fp)
    if best_fp is None:
        return None, float("nan")
    return best_fp, best_elpd


def metropolis_hastings_model_search(
    *,
    initial_config: dict,
    proposal_options: dict,
    data_norm: np.ndarray,
    n_iterations: int = 300,
    draws: int = 800,
    tune: int = 1200,
    nuts_backend: str = "pymc",
    chains: int = 2,
    cores: int | None = None,
    tau_mode: str = "marginalized",
    tau_lower: int = 2,
    tau_upper: int | None = None,
    rem_profile_params: dict | None = None,
    cache_fits: bool = True,
    seed: int | None = None,
    target_weights: dict[str, float] | None = None,
    verbose: bool = False,
    ic_criterion: str = "waic",
    quantile_window: int = 20,
    n_mh_chains: int = 1,
    run_ppc_for_best: bool = False,
    ppc_observed_data: np.ndarray | dict[str, np.ndarray] | None = None,
    ppc_num_pp_samples: int = 300,
    ppc_random_seed: int | None = None,
) -> dict[str, Any]:
    """Outer Metropolis-Hastings over discrete model configs; inner NUTS per accepted evaluation.

    Each iteration proposes one neighboring model, fits it with PyMC, and accepts/rejects
    using ``changepoint_log_target`` on ``score_changepoint_trace`` outputs (WAIC/LOO elpd
    when a PyMC ``model`` is available from the fit step).
    """
    tw = target_weights or {}
    n_mh_chains = max(1, int(n_mh_chains))

    def _final_log_target(chain_result: dict[str, Any]) -> float:
        ch = list(chain_result.get("chain") or [])
        if not ch:
            return float("-inf")
        return float(ch[-1].get("log_target", float("-inf")))

    def run_one_chain(seed_local: int | None) -> dict[str, Any]:
        rng = np.random.default_rng(seed_local)
        current = _clone_config(initial_config)
        if rem_profile_params is not None:
            current["rem_profile_params"] = dict(rem_profile_params)

        score_cache: dict[str, dict[str, Any]] = {}
        trace_cache: dict[str, Any] = {}
        model_cache: dict[str, Any] = {}
        group_data_cache: dict[str, dict] = {}
        rem_data_cache: dict[str, np.ndarray] = {}

        def ensure_data_for_config(cfg: dict) -> np.ndarray:
            """Return data_norm aligned with cfg['rem_profile_params'], with per-REM caching."""
            nonlocal data_norm
            rpp = cfg.get("rem_profile_params")
            if not rpp or _RUNTIME_LAST_EXPORT_CFG is None:
                return data_norm
            rem_key = json.dumps(
                {
                    "window_size_hours": int(rpp["window_size_hours"]),
                    "step_size_hours": int(rpp["step_size_hours"]),
                    "rem_stage": int(rpp["rem_stage"]),
                },
                sort_keys=True,
            )
            if rem_key in rem_data_cache:
                return rem_data_cache[rem_key]
            export_cfg = dict(_RUNTIME_LAST_EXPORT_CFG)
            export_cfg.update(
                {
                    "window_size_hours": int(rpp["window_size_hours"]),
                    "step_size_hours": int(rpp["step_size_hours"]),
                    "rem_stage": int(rpp["rem_stage"]),
                }
            )
            export_result = export_rem_profiles_10days_cached_only(**export_cfg)
            prep_cfg = _RUNTIME_LAST_PREPARE_CFG or {}
            prep_csv_path = export_result["paths"]["nanpad_output_csv"]
            prep = prepare_model_data(
                csv_path=prep_csv_path,
                bad_sample_indices=prep_cfg.get("bad_sample_indices"),
            )
            set_runtime_data_norm(prep["data_norm"])
            rem_data_cache[rem_key] = prep["data_norm"]
            return prep["data_norm"]

        chain_records: List[dict[str, Any]] = []
        fp0 = changepoint_model_config_fingerprint(current)
        data_work = ensure_data_for_config(current)
        gd0, tr0, sp0, m0 = _fit_config_once(
            current,
            data_norm=data_work,
            draws=draws,
            tune=tune,
            nuts_backend=nuts_backend,
            chains=chains,
            cores=cores,
            tau_mode=tau_mode,
            tau_lower=tau_lower,
            tau_upper=tau_upper,
            ic_criterion=ic_criterion,
        )
        score_cache[fp0] = sp0
        if cache_fits:
            trace_cache[fp0] = tr0
            model_cache[fp0] = m0
            group_data_cache[fp0] = gd0
        log_cur = changepoint_log_target(sp0, **tw)

        rec0: dict[str, Any] = {
            "iteration": 0,
            "fingerprint": fp0,
            "config": _clone_config(current),
            "accepted": True,
            "log_target": log_cur,
            "score": sp0,
        }
        chain_records.append(rec0)
        q10, q50, q90 = _running_log_target_quantiles(chain_records, quantile_window)
        chain_records[-1].update(
            {"log_target_q10": q10, "log_target_q50": q50, "log_target_q90": q90}
        )

        n_accept = 0
        for it in range(1, n_iterations + 1):
            proposed = propose_changepoint_model_config(current, proposal_options, rng)
            proposed.setdefault("tau_threshold", current.get("tau_threshold", 7.0))
            fp = changepoint_model_config_fingerprint(proposed)
            if cache_fits and fp in score_cache:
                sp_star = score_cache[fp]
                log_star = changepoint_log_target(sp_star, **tw)
            else:
                try:
                    dnorm = ensure_data_for_config(proposed)
                    gd_star, tr_star, sp_star, m_star = _fit_config_once(
                        proposed,
                        data_norm=dnorm,
                        draws=draws,
                        tune=tune,
                        nuts_backend=nuts_backend,
                        chains=chains,
                        cores=cores,
                        tau_mode=tau_mode,
                        tau_lower=tau_lower,
                        tau_upper=tau_upper,
                        ic_criterion=ic_criterion,
                    )
                    if cache_fits:
                        score_cache[fp] = sp_star
                        trace_cache[fp] = tr_star
                        model_cache[fp] = m_star
                        group_data_cache[fp] = gd_star
                except Exception as exc:
                    if verbose:
                        print(f"[MH iter {it}] proposal rejected (build/sample error): {exc}")
                    chain_records.append(
                        {
                            "iteration": it,
                            "fingerprint": fp,
                            "config": _clone_config(proposed),
                            "accepted": False,
                            "log_target": float("-inf"),
                            "score": None,
                            "error": str(exc),
                        }
                    )
                    q10, q50, q90 = _running_log_target_quantiles(chain_records, quantile_window)
                    chain_records[-1].update(
                        {"log_target_q10": q10, "log_target_q50": q50, "log_target_q90": q90}
                    )
                    continue
                log_star = changepoint_log_target(sp_star, **tw)

            log_alpha = log_star - log_cur
            if math.log(rng.random()) < log_alpha:
                current = _clone_config(proposed)
                log_cur = log_star
                n_accept += 1
                acc = True
            else:
                acc = False

            cur_fp = changepoint_model_config_fingerprint(current)
            chain_records.append(
                {
                    "iteration": it,
                    "fingerprint": cur_fp,
                    "config": _clone_config(current),
                    "accepted": acc,
                    "log_target": log_cur,
                    "score": score_cache.get(cur_fp),
                }
            )
            q10, q50, q90 = _running_log_target_quantiles(chain_records, quantile_window)
            chain_records[-1].update(
                {"log_target_q10": q10, "log_target_q50": q50, "log_target_q90": q90}
            )
            if verbose and it % 10 == 0:
                print(
                    f"[MH] iter={it} accept_rate~{n_accept / it:.3f} log_target={log_cur:.3f} "
                    f"log_target_q10/50/90={q10:.3f}/{q50:.3f}/{q90:.3f}"
                )

        return {
            "chain": chain_records,
            "acceptance_rate": n_accept / max(n_iterations, 1),
            "score_cache": score_cache,
            "trace_cache": trace_cache if cache_fits else {},
            "model_cache": model_cache if cache_fits else {},
            "group_data_cache": group_data_cache if cache_fits else {},
            "final_config": _clone_config(current),
        }

    chain_results: List[dict[str, Any]] = []
    for ci in range(n_mh_chains):
        seed_c = None if seed is None else int(seed) + ci
        chain_results.append(run_one_chain(seed_c))

    best = max(chain_results, key=_final_log_target)
    out = dict(best)
    best_fp, best_elpd = _best_fingerprint_by_elpd(out["score_cache"])
    out["best_fingerprint"] = best_fp
    out["best_elpd"] = best_elpd

    if n_mh_chains > 1:
        finals = [_final_log_target(cr) for cr in chain_results]
        out["mh_chain_results"] = chain_results
        out["final_log_targets"] = finals
        fa = np.asarray([x for x in finals if math.isfinite(x)], dtype=float)
        out["final_log_target_std"] = float(np.std(fa)) if fa.size > 1 else 0.0

    if run_ppc_for_best:
        best_trace = None
        best_model = None
        if best_fp is not None:
            best_trace = (out.get("trace_cache") or {}).get(best_fp)
            best_model = (out.get("model_cache") or {}).get(best_fp)
        if best_trace is not None and best_model is not None:
            out["best_model_ppc"] = plot_posterior_predictive_check(
                best_trace,
                best_model,
                observed_data=ppc_observed_data,
                num_pp_samples=ppc_num_pp_samples,
                random_seed=ppc_random_seed,
            )
        else:
            warnings.warn(
                "metropolis_hastings_model_search: run_ppc_for_best=True but best trace/model "
                "is unavailable (likely cache_fits=False or no best fingerprint).",
                UserWarning,
                stacklevel=2,
            )
    return out


def summarize_model_search(search_result: dict) -> dict[str, Any]:
    """Aggregate MH chain: model visit counts, best log-target seen, feature/likelihood visit frequencies."""
    chain = list(search_result.get("chain") or [])
    if not chain:
        return {
            "model_visit_counts": {},
            "top_fingerprints_by_log_target": [],
            "feature_visit_freq": {},
            "likelihood_visit_freq": {},
            "top_fingerprints_by_elpd": [],
            "elpd_by_fingerprint": {},
            "final_log_target_std": float("nan"),
            "final_log_targets": [],
        }

    from collections import Counter

    visit_fp: List[str] = []
    w_feat: dict[str, float] = {}
    w_like: dict[str, dict[str, float]] = {}
    log_targets: dict[str, float] = {}
    elpd_by_fp: dict[str, float] = {}

    for rec in chain:
        fp = rec.get("fingerprint")
        if not fp:
            continue
        fp = str(fp)
        visit_fp.append(fp)
        sc = rec.get("score") or {}
        lt = float(rec.get("log_target", float("-inf")))
        if math.isfinite(lt):
            log_targets[fp] = max(log_targets.get(fp, float("-inf")), lt)
        elp = sc.get("elpd")
        if elp is not None and math.isfinite(float(elp)):
            efv = float(elp)
            elpd_by_fp[fp] = max(elpd_by_fp.get(fp, float("-inf")), efv)
        for feat in sc.get("active_features") or []:
            w_feat[feat] = w_feat.get(feat, 0.0) + 1.0
        likes = sc.get("likelihoods") or {}
        for feat, lk in likes.items():
            w_like.setdefault(feat, {})
            w_like[feat][lk] = w_like[feat].get(lk, 0.0) + 1.0

    score_cache = search_result.get("score_cache") or {}
    for fp, sc in score_cache.items():
        if not isinstance(sc, dict):
            continue
        fp = str(fp)
        elp = sc.get("elpd")
        if elp is None or not math.isfinite(float(elp)):
            continue
        efv = float(elp)
        elpd_by_fp[fp] = max(elpd_by_fp.get(fp, float("-inf")), efv)

    counts = Counter(visit_fp)
    top = sorted(log_targets.keys(), key=lambda f: log_targets[f], reverse=True)[:15]
    top_elpd = sorted(elpd_by_fp.keys(), key=lambda f: elpd_by_fp[f], reverse=True)[:15]

    def _renorm(d: dict[str, float]) -> dict[str, float]:
        s = sum(d.values()) or 1.0
        return {k: v / s for k, v in sorted(d.items(), key=lambda kv: -kv[1])}

    like_renorm = {feat: _renorm(vs) for feat, vs in w_like.items()}
    out: dict[str, Any] = {
        "model_visit_counts": dict(counts.most_common(25)),
        "top_fingerprints_by_log_target": top,
        "log_target_by_fingerprint": log_targets,
        "feature_visit_freq": _renorm(w_feat),
        "likelihood_visit_freq": like_renorm,
        "mean_acceptance_indicator": float(np.mean([1.0 if r.get("accepted") else 0.0 for r in chain[1:]]))
        if len(chain) > 1
        else 0.0,
        "top_fingerprints_by_elpd": top_elpd,
        "elpd_by_fingerprint": elpd_by_fp,
        "final_log_target_std": float(search_result.get("final_log_target_std", float("nan"))),
        "final_log_targets": list(search_result.get("final_log_targets") or []),
    }
    return out


def plot_model_search_results(
    search_result: dict,
    summary: dict[str, Any] | None = None,
    *,
    title: str = "Metropolis-Hastings model search",
) -> None:
    """Plot MH chain: log-target, running acceptance, P(tau>thr) / MAP tau, feature and likelihood visit shares."""
    chain = list(search_result.get("chain") or [])
    if not chain:
        print("No MH chain to plot.")
        return

    summary = summary if summary is not None else summarize_model_search(search_result)

    iterations = np.array([int(r["iteration"]) for r in chain], dtype=float)
    log_targets = np.array([float(r.get("log_target", float("nan"))) for r in chain], dtype=float)
    accepted = np.array([bool(r.get("accepted", False)) for r in chain], dtype=bool)
    q10_arr = np.array([float(r.get("log_target_q10", float("nan"))) for r in chain], dtype=float)
    q50_arr = np.array([float(r.get("log_target_q50", float("nan"))) for r in chain], dtype=float)
    q90_arr = np.array([float(r.get("log_target_q90", float("nan"))) for r in chain], dtype=float)

    thr = float(chain[0].get("config", {}).get("tau_threshold", 7.0))
    p_gt: List[float] = []
    map_tau: List[float] = []
    bfmi_arr: List[float] = []
    ndiv_arr: List[float] = []
    for r in chain:
        sc = r.get("score")
        if isinstance(sc, dict):
            p_gt.append(float(sc.get("p_tau_gt_threshold", float("nan"))))
            map_tau.append(float(sc.get("map_tau", float("nan"))))
            b = sc.get("bfmi", sc.get("bfmi_approx"))
            if b is not None and math.isfinite(float(b)):
                bfmi_arr.append(float(b))
            else:
                bfmi_arr.append(float("nan"))
            ndiv_arr.append(float(sc.get("n_divergences", 0)))
        else:
            p_gt.append(float("nan"))
            map_tau.append(float("nan"))
            bfmi_arr.append(float("nan"))
            ndiv_arr.append(float("nan"))
    p_gt_arr = np.asarray(p_gt, dtype=float)
    map_tau_arr = np.asarray(map_tau, dtype=float)
    bfmi_plot = np.asarray(bfmi_arr, dtype=float)
    ndiv_plot = np.asarray(ndiv_arr, dtype=float)

    full_title = title
    flt = summary.get("final_log_targets") or []
    flstd = summary.get("final_log_target_std")
    if len(flt) > 1:
        try:
            sd = float(flstd)
            if math.isfinite(sd):
                full_title = f"{title} (std of final log-target across {len(flt)} MH chains: {sd:.4g})"
        except (TypeError, ValueError):
            pass

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax_lt = axes[0, 0]
    if np.any(np.isfinite(q10_arr)) and np.any(np.isfinite(q90_arr)):
        ax_lt.fill_between(
            iterations,
            q10_arr,
            q90_arr,
            color="#348ABD",
            alpha=0.22,
            label="log-target q10–q90 (rolling)",
        )
        ax_lt.plot(
            iterations,
            q50_arr,
            "--",
            color="#1f77b4",
            lw=1.2,
            alpha=0.85,
            label="rolling median",
        )
    ax_lt.plot(iterations, log_targets, "o-", color="#E24A33", ms=5, lw=1.2, label="current log-target")
    ax_lt.set_xlabel("MH iteration")
    ax_lt.set_ylabel("log target (current state)")
    ax_lt.set_title("Log-target along chain (+ rolling quantiles)")
    ax_lt.grid(alpha=0.3)
    ax_lt.legend(loc="best", fontsize=8)

    if len(chain) > 1:
        acc_float = accepted[1:].astype(float)
        run_mean = np.cumsum(acc_float) / np.arange(1, len(acc_float) + 1, dtype=float)
        axes[0, 1].plot(iterations[1:], run_mean, color="#E24A33", lw=1.5)
        axes[0, 1].scatter(iterations[1:], acc_float, c=["#2ca02c" if a else "#7f7f7f" for a in accepted[1:]], s=22, zorder=3)
        axes[0, 1].set_xlabel("MH iteration")
        axes[0, 1].set_ylabel("cumulative P(accept proposal)")
        axes[0, 1].set_title("Running acceptance rate (green=accepted)")
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].grid(alpha=0.3)
    else:
        axes[0, 1].axis("off")

    axb = axes[1, 0]
    axb.plot(iterations, p_gt_arr, "s-", color="#2ca02c", ms=4, lw=1, label=rf"$P(\tau > {thr:g})$")
    axb.set_xlabel("MH iteration")
    axb.set_ylabel(rf"$P(\tau > {thr:g})$")
    axb.set_ylim(-0.05, 1.05)
    axb.legend(loc="upper left")
    axb.grid(alpha=0.3)

    ax2 = axb.twinx()
    ax2.plot(iterations, map_tau_arr, "D--", color="#9467bd", ms=4, lw=1, alpha=0.85, label="MAP " + r"$\tau$")
    ax2.set_ylabel("MAP " + r"$\tau$" + " (from posterior over support)")
    ax2.legend(loc="upper right")

    axb.set_title(r"$\tau$ signal along chain (marginalized / discrete)")

    ax_s = axes[1, 1]
    ax_s.plot(iterations, bfmi_plot, "o-", color="#8c564b", ms=4, lw=1.1, label="BFMI")
    ax_s.axhline(0.3, color="gray", ls="--", lw=0.9, label="BFMI=0.3")
    ax_s.set_xlabel("MH iteration")
    ax_s.set_ylabel("BFMI")
    ax_s.set_title("Sampler diagnostics (current model state)")
    ax_s.grid(alpha=0.3)
    ax_sd = ax_s.twinx()
    ax_sd.bar(
        iterations,
        ndiv_plot,
        color="#E24A33",
        width=0.8,
        alpha=0.9,
        label="NUTS divergences / fit",
    )
    ax_sd.set_ylabel("divergences")
    ax_s.legend(loc="upper left", fontsize=8)
    ax_sd.legend(loc="upper right", fontsize=8)

    plt.suptitle(full_title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

    feat = summary.get("feature_visit_freq") or {}
    if feat:
        names = list(feat.keys())
        vals = [float(feat[k]) for k in names]
        y_pos = np.arange(len(names))
        fig_feat, axf = plt.subplots(figsize=(6.5, max(2.5, 0.35 * len(names))))
        axf.barh(y_pos, vals, color="#A60628", height=0.65)
        axf.set_yticks(y_pos)
        axf.set_yticklabels(names)
        axf.set_xlabel("visit share")
        axf.set_title("Feature metrics in model (visit share)")
        axf.set_xlim(0, 1.05)
        axf.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    lk = summary.get("likelihood_visit_freq") or {}
    if not lk:
        return

    metrics_sorted = sorted(lk.keys())
    n_m = len(metrics_sorted)
    fig2, axes2 = plt.subplots(1, n_m, figsize=(max(4.0 * n_m, 5), 3.2), squeeze=False)
    for i, metric in enumerate(metrics_sorted):
        ax = axes2[0, i]
        dists = lk[metric]
        labs = list(dists.keys())
        vs = [float(dists[lab]) for lab in labs]
        ax.bar(labs, vs, color="#7A68A6", edgecolor="black", linewidth=0.5)
        ax.set_title(metric)
        ax.set_ylabel("visit share")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Likelihood family (visit share per metric)", fontsize=11, y=1.08)
    plt.tight_layout()
    plt.show()


def run_variant(
    title: str,
    *,
    n_chunks: int,
    feature_selection,
    parameter_selection: dict | None = None,
    rem_profile_params: dict | None = None,
    draws: int = 4000,
    tune: int = 2000,
    nuts_backend: str = "pymc",
    chains: int = 4,
    cores: int | None = None,
    tau_mode: str = "discrete",
    tau_lower: int = 2,
    tau_upper: int | None = None,
    data_norm: np.ndarray | None = None,
    plot_likelihood_profiles: bool = True,
    likelihood_profile_grid_size: int = 300,
    return_likelihood_profiles: bool = False,
):
    """Full scenario run: features -> model -> MCMC -> summary -> plots."""
    normalized_rem_profile_params: dict[str, int] | None = None
    if rem_profile_params is not None:
        required_keys = {"window_size_hours", "step_size_hours", "rem_stage"}
        missing = sorted(required_keys - set(rem_profile_params))
        if missing:
            raise ValueError(
                "rem_profile_params is missing required keys: "
                f"{missing}. Expected keys: {sorted(required_keys)}"
            )
        w, s, r = _normalize_rem_profile_params(
            window_size_hours=rem_profile_params["window_size_hours"],
            step_size_hours=rem_profile_params["step_size_hours"],
            rem_stage=rem_profile_params["rem_stage"],
        )
        normalized_rem_profile_params = {
            "window_size_hours": w,
            "step_size_hours": s,
            "rem_stage": r,
        }

    backend = str(nuts_backend).strip().lower()
    if backend in {"numpyro", "blackjax"} and str(tau_mode).strip().lower() == "discrete":
        print(
            "JAX NUTS backend with discrete tau is unsupported; using tau_mode='marginalized'."
        )
        tau_mode = "marginalized"

    data_for_run = data_norm if data_norm is not None else _RUNTIME_DATA_NORM
    if normalized_rem_profile_params is not None and data_norm is None:
        if _RUNTIME_LAST_EXPORT_CFG is None:
            print(
                "rem_profile_params were provided, but no prior export config is available. "
                "Using existing runtime data without recalculation."
            )
        else:
            export_cfg = dict(_RUNTIME_LAST_EXPORT_CFG)
            export_cfg.update(normalized_rem_profile_params)
            print("Recalculating REM profiles with updated rem_profile_params...")
            export_result = export_rem_profiles_10days_cached_only(**export_cfg)

            prep_cfg = _RUNTIME_LAST_PREPARE_CFG or {}
            prep_csv_path = export_result["paths"]["nanpad_output_csv"]
            prep_bad_indices = prep_cfg.get("bad_sample_indices", None)
            prep = prepare_model_data(
                csv_path=prep_csv_path,
                bad_sample_indices=prep_bad_indices,
            )
            set_runtime_data_norm(prep["data_norm"])
            data_for_run = prep["data_norm"]
            print(
                "Recalculation complete: "
                f"data_norm shape={data_for_run.shape}, csv_path={prep_csv_path!r}"
            )

    if data_for_run is None:
        raise ValueError("data_norm is not set. Pass data_norm=... or call set_runtime_data_norm(...).")

    print("=" * 90)
    print(f"Сценарий: {title}")
    print(f"  n_chunks={n_chunks}")
    print(f"  feature_selection={feature_selection!r}")
    if parameter_selection is not None:
        print(f"  parameter_selection={parameter_selection!r}")
    if normalized_rem_profile_params is not None:
        print(f"  rem_profile_params={normalized_rem_profile_params!r}")
        print(
            "  note: rem_profile_params trigger recalculation only when runtime export "
            "and prepare configs are available (or when data_norm is passed explicitly)."
        )
    print(f"  nuts_backend={nuts_backend!r}")
    print(f"  tau_mode={tau_mode!r}")
    print("=" * 90)

    group_data = build_group_data(
        data_for_run,
        n_chunks=n_chunks,
        feature_selection=feature_selection,
    )
    for group_name, features in group_data.items():
        for feat_name, df in features.items():
            print(f"Группа '{group_name}', признак '{feat_name}': форма {df.shape}")
            print(f"Первые 2 строки ({feat_name}):")
            print(df.head(2).to_string(index=False))
            print()

    model = build_changepoint_model(
        group_data,
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        parameter_selection=parameter_selection,
        tau_mode=tau_mode,
    )
    trace = sample_model(
        model,
        draws=draws,
        tune=tune,
        nuts_backend=nuts_backend,
        chains=chains,
        cores=cores,
    )
    available_vars = _available_varnames(trace)
    trace_vars = []
    summary_vars = []
    if "tau" in available_vars:
        trace_vars.append("tau")
        summary_vars.append("tau")
    if "tau_mean" in available_vars:
        trace_vars.append("tau_mean")
        summary_vars.append("tau_mean")

    for group_name, features in group_data.items():
        for feat_name in features.keys():
            for param_name in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in available_vars and p2 in available_vars:
                    trace_vars.extend([p1, p2])
                    summary_vars.extend([p1, p2])
            nu_name = f"nu_{group_name}_{feat_name}"
            if nu_name in available_vars:
                summary_vars.append(nu_name)

    if summary_vars:
        summary = summary_from_trace(trace, summary_vars)
        print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())
    else:
        summary = pd.DataFrame()
        print("Summary: no scalar parameters selected for compact table.")

    diverging = _sampler_stat(trace, "diverging")
    n_div = int(np.asarray(diverging).sum())
    print(f"Дивергенции: {n_div}")

    energy = np.asarray(_sampler_stat(trace, "energy"), dtype=float).reshape(-1)
    if energy.size > 1 and np.var(energy) > 0:
        bfmi = float(np.mean(np.diff(energy) ** 2) / np.var(energy))
        print(f"BFMI (approx): {bfmi:.3f}")
    else:
        print("BFMI (approx): недостаточно данных.")

    support, probs = tau_probabilities(trace)
    print("Вероятности tau:")
    for k, p in zip(support, probs):
        print(f"  P(tau={k}) = {p:.3f}")
    map_idx = int(np.argmax(probs))
    print(f"MAP tau: {int(support[map_idx])}, концентрация: {float(probs[map_idx]):.3f}")

    plot_trace_and_tau(trace, trace_vars, title_prefix=title)
    plot_posteriors_like_script(trace, group_data=group_data, title_prefix=title)
    likelihood_profiles = feature_likelihood_profiles(
        trace,
        group_data=group_data,
        parameter_selection=parameter_selection,
        grid_size=likelihood_profile_grid_size,
        plot=plot_likelihood_profiles,
    )
    if return_likelihood_profiles:
        return trace, summary, likelihood_profiles
    return trace, summary


def plot_posterior_predictive_check(
    trace,
    model,
    observed_data: np.ndarray | dict[str, np.ndarray] | None = None,
    *,
    num_pp_samples: int = 300,
    random_seed: int | None = None,
) -> Any:
    """Posterior predictive check for changepoint models.

    Draws posterior predictive samples and overlays predictive bands with observed values.
    Falls back to ``az.plot_ppc`` when observed data shape cannot be inferred robustly.
    """
    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["y_like"],
            random_seed=random_seed,
            return_inferencedata=True,
            extend_inferencedata=False,
            predictions=False,
        )

    y_pp = None
    try:
        y_pp = np.asarray(ppc.posterior_predictive["y_like"], dtype=float)
    except Exception:
        y_pp = None

    y_obs = observed_data
    if y_obs is None and hasattr(model, "observed_RVs") and model.observed_RVs:
        try:
            y_obs = np.asarray(model.observed_RVs[0].eval(), dtype=float)
        except Exception:
            y_obs = None

    if y_pp is None:
        az.plot_ppc(ppc, num_pp_samples=int(max(20, num_pp_samples)))
        plt.tight_layout()
        plt.show()
        return ppc

    # y_pp dims are typically (chain, draw, *obs_shape). Collapse chain/draw and avg over features.
    y_samples = y_pp.reshape((-1,) + y_pp.shape[2:])
    if y_samples.ndim == 1:
        y_samples = y_samples[:, np.newaxis]
    if y_samples.ndim >= 3:
        y_samples = y_samples.mean(axis=tuple(range(2, y_samples.ndim)))
    if y_samples.ndim == 1:
        y_samples = y_samples[:, np.newaxis]

    q05 = np.quantile(y_samples, 0.05, axis=0)
    q50 = np.quantile(y_samples, 0.50, axis=0)
    q95 = np.quantile(y_samples, 0.95, axis=0)
    x = np.arange(q50.shape[0], dtype=int)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x, q05, q95, color="#1f77b4", alpha=0.22, label="posterior predictive 90% band")
    ax.plot(x, q50, color="#1f77b4", lw=1.6, label="posterior predictive median")

    if y_obs is not None:
        y_obs_arr = np.asarray(y_obs, dtype=float)
        if y_obs_arr.ndim > 1:
            y_obs_arr = y_obs_arr.mean(axis=tuple(range(1, y_obs_arr.ndim)))
        if y_obs_arr.shape[0] == q50.shape[0]:
            ax.plot(x, y_obs_arr, color="#E24A33", lw=1.4, alpha=0.9, label="observed")
        else:
            warnings.warn(
                "plot_posterior_predictive_check: observed_data length does not match predictive shape; "
                "showing predictive envelope only.",
                UserWarning,
                stacklevel=2,
            )

    ax.set_xlabel("observation index")
    ax.set_ylabel("value")
    ax.set_title("Posterior predictive check")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return ppc


__all__ = [
    "DEFAULT_EVENTS_10D",
    "DEFAULT_S3_CONFIG",
    "export_rem_profiles_10days_cached_only",
    "maxmin_scale",
    "load_and_normalize",
    "compute_chunk_feature_map",
    "compute_concat_chunk_feature_map",
    "compute_chunk_features",
    "prepare_variant_data",
    "build_group_data",
    "prepare_model_data",
    "set_runtime_data_norm",
    "FEATURE_SELECTION_PRESETS",
    "parameter_selection_with_g_prior",
    "build_changepoint_model",
    "sample_model",
    "summary_from_trace",
    "tau_probabilities",
    "feature_likelihood_profiles",
    "plot_trace_and_tau",
    "plot_posteriors_like_script",
    "changepoint_model_config_fingerprint",
    "score_changepoint_trace",
    "changepoint_log_target",
    "propose_changepoint_model_config",
    "metropolis_hastings_model_search",
    "summarize_model_search",
    "plot_model_search_results",
    "plot_posterior_predictive_check",
    "run_variant",
]
