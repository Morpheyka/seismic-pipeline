"""REM profile export from cached hypnograms."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .date_utils import parse_event_date
from .hypnogram_cache_manager import HypnogramCacheManagerYt
from .rem_profile_calculator import REMProfileCalculatorYt
from seismic_pipeline.config.changepoint_defaults import (
    s3_config_from_env,
    validate_rem_profile_params,
)
from seismic_pipeline.features.runtime import (
    set_runtime_export_cfg,
    _RUNTIME_LAST_EXPORT_CFG,
)

def _normalize_rem_profile_params(
    *,
    window_size_hours: int,
    step_size_hours: int,
    rem_stage: int,
) -> tuple[int, int, int]:
    """Validate and normalize legacy REM profile generation parameters."""
    validated = validate_rem_profile_params(
        {
            "window_size_hours": window_size_hours,
            "step_size_hours": step_size_hours,
            "rem_stage": rem_stage,
        }
    )
    return (
        int(window_size_hours),
        int(step_size_hours),
        int(validated["rem_stage"]),
    )
def _parse_event_date(date_str: str) -> datetime:
    return parse_event_date(date_str)
def _build_nday_inputs(
    events: Iterable[Dict[str, str]],
    window_days: int = 10,
) -> List[Dict[str, object]]:
    if window_days <= 0:
        raise ValueError(f"window_days must be > 0, got {window_days}")

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
                for offset in range(window_days, 0, -1)
            ]
        else:
            # Reverse order for post-event window: +N, +(N-1), ... +1.
            window_dates = [
                (dt + timedelta(days=offset)).strftime("%Y_%m_%d")
                for offset in range(window_days, 0, -1)
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
def _build_10day_inputs(events: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    """Backward-compatible 10-day alias."""
    return _build_nday_inputs(events, window_days=10)
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
        if cache_manager.cache_hypnogram(rat_id, date, source="s3"):
            cached.append((rat_id, date))
            continue
        if cache_manager.cache_hypnogram(rat_id, date, source="local"):
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
    exported_day_lengths: Dict[int, List[int]] | None = None,
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
        day_lens = exported_day_lengths.get(idx) if exported_day_lengths else None
        day_lens_str = ";".join(str(int(x)) for x in day_lens) if day_lens else ""
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
                "day_profile_lengths": day_lens_str,
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
    n_points_per_day: int | None = None,
    overlap: float | None = None,
    window_days: int = 10,
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
    """Export REM profile vectors from cached/local/S3 hypnograms only.

    Per-day profiles use fixed ``n_points_per_day`` with ``overlap`` (default derived
    from legacy ``window_size_hours`` / ``step_size_hours`` via conversion).

    When ``concat_hypnogram_for_event=True``, the concatenated hypnogram is processed
    with the legacy hour-based sliding window (not fixed-N per day).
    """
    global _RUNTIME_LAST_EXPORT_CFG

    if n_points_per_day is not None:
        rem_params = validate_rem_profile_params(
            {
                "n_points_per_day": n_points_per_day,
                "overlap": overlap if overlap is not None else 0.0,
                "rem_stage": rem_stage,
            }
        )
    else:
        rem_params = validate_rem_profile_params(
            {
                "window_size_hours": window_size_hours,
                "step_size_hours": step_size_hours,
                "rem_stage": rem_stage,
            }
        )

    n_pts = int(rem_params["n_points_per_day"])
    ov = float(rem_params["overlap"])
    rem_stage = int(rem_params["rem_stage"])

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
        "n_points_per_day": n_pts,
        "overlap": ov,
        "window_days": window_days,
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
    set_runtime_export_cfg(_RUNTIME_LAST_EXPORT_CFG)

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, output_csv)
    output_csv_nanpad_path = os.path.join(output_dir, output_csv_nanpad)
    metadata_csv_path = os.path.join(output_dir, metadata_csv)
    summary_json_path = os.path.join(output_dir, "samples_10days_summary.json")

    cache_manager = HypnogramCacheManagerYt(
        local_cache_dir=local_hypnogram_cache_dir,
        local_data_root=local_data_root,
        s3_config=s3_config or s3_config_from_env(),
        s3_rat_bucket=s3_rat_bucket,
        s3_temp_bucket=s3_temp_bucket,
        # Keep fallback roots enabled so notebook configs with '/mnt/wd/rat'
        # still resolve to '~/mnt/wd/rat' when mounted there.
        allow_local_root_fallback=True,
    )

    rows = _build_nday_inputs(events, window_days=window_days)
    cached_pairs, missing_pairs = _cache_needed_dates(cache_manager, rows)
    print(f"Total required (rat,date) pairs: {len(cached_pairs) + len(missing_pairs)}")
    print(f"Initially cached/available: {len(cached_pairs)}")
    print(f"Initially missing: {len(missing_pairs)}")
    print(
        "REM profile params: "
        f"n_points_per_day={n_pts}, overlap={ov}, rem_stage={rem_stage} "
        f"(legacy window_size_hours={window_size_hours}, step_size_hours={step_size_hours})"
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
        n_points_per_day=n_pts if not concat_hypnogram_for_event else None,
        overlap=ov,
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
        day_lengths_list = None
    else:
        raw_features, valid_indices, day_lengths_list = rem_calc._calculate_features_for_X(
            export_rows,
            return_day_lengths=True,
        )

    if not raw_features:
        raise ValueError(
            "No REM profile vectors were produced from the selected rows. "
            "Check missing data and cache contents."
        )

    if concat_hypnogram_for_event:
        padded_vector_length = int(max(len(v) for v in raw_features))
    else:
        padded_vector_length = int(window_days * n_pts)
    exported_original_indices: List[int] = []
    true_vector_lengths: List[int] = [0] * len(rows)
    padded_zero_rows: List[np.ndarray] = []
    padded_nan_rows: List[np.ndarray] = []
    matrix_day_lengths: List[List[int]] = []

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
        if day_lengths_list is not None:
            matrix_day_lengths.append(list(day_lengths_list[k]))
        else:
            matrix_day_lengths.append([])

    X_zero = np.vstack(padded_zero_rows)
    X_nan = np.vstack(padded_nan_rows)

    day_lengths_csv_path = os.path.join(
        output_dir,
        output_csv_nanpad.replace(".csv", "_day_lengths.csv"),
    )
    if matrix_day_lengths and any(matrix_day_lengths):
        pd.DataFrame(
            {
                "day_profile_lengths": [
                    ";".join(str(int(x)) for x in lengths) for lengths in matrix_day_lengths
                ]
            }
        ).to_csv(day_lengths_csv_path, index=False)

    exported_day_lengths: Dict[int, List[int]] = {}
    if day_lengths_list is not None:
        for k, _features in enumerate(raw_features):
            export_row_idx = int(valid_indices[k])
            original_row_idx = int(kept_indices_array[export_row_idx])
            exported_day_lengths[original_row_idx] = list(day_lengths_list[k])

    pd.DataFrame(X_zero).to_csv(output_csv_path, index=False, header=False)
    pd.DataFrame(X_nan).to_csv(output_csv_nanpad_path, index=False, header=False)

    metadata_df = _build_metadata(
        rows=rows,
        missing_pairs=missing_pairs,
        exported_original_indices=np.array(exported_original_indices, dtype=int),
        true_vector_lengths=true_vector_lengths,
        padded_vector_length=padded_vector_length,
        exported_day_lengths=exported_day_lengths or None,
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
        "n_points_per_day": n_pts,
        "overlap": ov,
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
