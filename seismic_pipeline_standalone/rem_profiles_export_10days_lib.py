"""
Cached-only 10-day REM profile export utilities.

This module provides a notebook- and CLI-friendly API for exporting samples from
already available hypnograms (cache/local/S3 temp). It does not include any
quality-model or auto-hypnogram-from-dat logic.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

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
    return {
        "mean": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 1.5},
            "sigma_prior": {"dist": "halfnormal", "sigma": 1.0},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
        },
        "range": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -0.3, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.7},
        },
        "std": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -0.7, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.7},
        },
        "skewness": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 2.5},
            "sigma_prior": {"dist": "halfstudentt", "nu": 4.0, "sigma": 1.5},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
        },
        "kurtosis": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 3.0},
            "sigma_prior": {"dist": "halfstudentt", "nu": 4.0, "sigma": 2.0},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
        },
    }


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
    """Build PyMC changepoint model for group_data features."""
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
                    mu_1 = _build_prior(
                        f"mu_{group_name}_{feat_name}_1",
                        spec.get("mu_prior", {"dist": "normal", "mu": 0.0, "sigma": 2.0}),
                    )
                    mu_2 = _build_prior(
                        f"mu_{group_name}_{feat_name}_2",
                        spec.get("mu_prior", {"dist": "normal", "mu": 0.0, "sigma": 2.0}),
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
    "build_changepoint_model",
    "sample_model",
    "summary_from_trace",
    "tau_probabilities",
    "feature_likelihood_profiles",
    "plot_trace_and_tau",
    "plot_posteriors_like_script",
    "run_variant",
]
