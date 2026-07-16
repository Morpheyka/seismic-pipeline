#!/usr/bin/env python3
"""Build daily risk dataset (next-24h event) from cached 8-day profiles.

This script creates one row per rat-day and computes only causal features:
all features for day t use days <= t within the same 8-day window.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Iterable


@dataclass(frozen=True)
class DailySample:
    rat_id: str
    sample_date: date
    source_event_date: date
    day_idx: int
    label_next24h: int
    is_quiet_control: int
    group_event_date: date


ARTIFACT_DAYS_BY_KEY: dict[tuple[str, str, str], set[int]] = {
    ("R2", "2022-11-07", "before"): {0},
    ("R2", "2023-05-03", "before"): {2},
    ("R3", "2023-05-03", "before"): {2},
    ("R2", "2023-04-21", "after_reversed"): {3},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build risk dataset for next-24h event prediction.")
    parser.add_argument(
        "--profiles-dir",
        default="./run_output_old_vs_current_exhaustive/profile_plots_8days_data",
        help="Directory with samples_10days_nanpad.csv and samples_10days_metadata.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="./run_output_risk_model",
        help="Output directory for daily dataset and metadata.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=8,
        help="Number of days per profile window.",
    )
    parser.add_argument(
        "--quiet-buffer-days",
        type=int,
        default=8,
        help="Minimum absolute distance (days) from any event date for quiet controls.",
    )
    parser.add_argument(
        "--quiet-ratio",
        type=float,
        default=2.0,
        help="Target quiet controls per positive sample.",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rows.append([float(x) for x in row])
    return rows


def _read_exported_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    exported = [r for r in all_rows if str(r.get("exported", "")).strip().lower() == "true"]
    return exported


def _normalize_minus1_plus1(row: list[float]) -> list[float]:
    lo = min(row)
    hi = max(row)
    if hi == lo:
        return [0.0 for _ in row]
    return [2.0 * (x - lo) / (hi - lo) - 1.0 for x in row]


def _daily_profiles(norm_row: list[float], window_days: int) -> list[list[float]]:
    n_per_day = len(norm_row) // window_days
    if n_per_day * window_days != len(norm_row):
        raise ValueError(f"vector length {len(norm_row)} is not divisible by window_days={window_days}")
    return [norm_row[d * n_per_day : (d + 1) * n_per_day] for d in range(window_days)]


def _split_half(day_values: list[float]) -> tuple[list[float], list[float]]:
    mid = len(day_values) // 2
    return day_values[:mid], day_values[mid:]


def _slope_last(values: list[float], lookback: int = 3) -> float:
    vals = [v for v in values[-lookback:] if math.isfinite(v)]
    if len(vals) < 2:
        return math.nan
    x = list(range(len(vals)))
    x_bar = mean(x)
    y_bar = mean(vals)
    num = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, vals))
    den = sum((xi - x_bar) ** 2 for xi in x)
    return num / den if den > 0 else 0.0


def _pairwise_dispersion(days: list[list[float]], metric: str) -> float:
    if len(days) < 2:
        return math.nan
    vals: list[float] = []
    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            diff = [a - b for a, b in zip(days[i], days[j])]
            if metric == "l1":
                vals.append(sum(abs(x) for x in diff))
            elif metric == "l2":
                vals.append(math.sqrt(sum(x * x for x in diff)))
            else:
                raise ValueError(metric)
    return mean(vals)


def _distance_to_ref(day_values: list[float], ref_values: list[float], metric: str) -> float:
    diff = [a - b for a, b in zip(day_values, ref_values)]
    if metric == "l1":
        return sum(abs(x) for x in diff)
    if metric == "l2":
        return math.sqrt(sum(x * x for x in diff))
    raise ValueError(metric)


def _fmean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def _safe_std(values: list[float]) -> float:
    if not values:
        return math.nan
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _rolling_mean(values: list[float], k: int) -> float:
    if not values:
        return math.nan
    return mean(values[-k:])


def _build_day_level_features(
    daily: list[list[float]],
    day_idx: int,
    excluded_days: set[int],
) -> dict[str, float]:
    """Causal features for day_idx using only days <= day_idx and non-artifact days."""
    cur = daily[day_idx]
    first_half, second_half = _split_half(cur)

    hist_days_idx = [d for d in range(day_idx + 1) if d not in excluded_days]
    hist_days = [daily[d] for d in hist_days_idx]
    hist_means = [mean(d) for d in hist_days]
    hist_ranges = [max(d) - min(d) for d in hist_days]
    hist_stds = [_safe_std(d) for d in hist_days]

    cur_mean = mean(cur)
    cur_range = max(cur) - min(cur)
    cur_std = _safe_std(cur)

    features: dict[str, float] = {
        "mean_24h": cur_mean,
        "range_24h": cur_range,
        "std_24h": cur_std,
        "rem_share_24h": (cur_mean + 1.0) / 2.0,
        "mean_48h": _rolling_mean(hist_means, 2),
        "mean_72h": _rolling_mean(hist_means, 3),
        "range_48h": _rolling_mean(hist_ranges, 2),
        "range_72h": _rolling_mean(hist_ranges, 3),
        "std_48h": _rolling_mean(hist_stds, 2),
        "std_72h": _rolling_mean(hist_stds, 3),
        "mean_first_half": mean(first_half),
        "mean_second_half": mean(second_half),
        "range_first_half": max(first_half) - min(first_half),
        "range_second_half": max(second_half) - min(second_half),
        "trend_mean_last3": _slope_last(hist_means, lookback=3),
        "trend_range_last3": _slope_last(hist_ranges, lookback=3),
        "trend_std_last3": _slope_last(hist_stds, lookback=3),
    }

    # Deltas vs previous valid day.
    prev_days_idx = [d for d in range(day_idx) if d not in excluded_days]
    if prev_days_idx:
        prev = daily[prev_days_idx[-1]]
        features["delta_mean_1d"] = cur_mean - mean(prev)
        features["delta_range_1d"] = cur_range - (max(prev) - min(prev))
        diff = [a - b for a, b in zip(cur, prev)]
        features["shape_l1_lag1"] = sum(abs(x) for x in diff)
        features["shape_l2_lag1"] = math.sqrt(sum(x * x for x in diff))
    else:
        features["delta_mean_1d"] = math.nan
        features["delta_range_1d"] = math.nan
        features["shape_l1_lag1"] = math.nan
        features["shape_l2_lag1"] = math.nan

    # Dispersion among last 3 valid days (including current if valid).
    last3_idx = hist_days_idx[-3:]
    last3_days = [daily[d] for d in last3_idx]
    features["pairwise_l1_disp_last3"] = _pairwise_dispersion(last3_days, "l1")
    features["pairwise_l2_disp_last3"] = _pairwise_dispersion(last3_days, "l2")

    # Cross-fitted anomaly: current day vs median profile of previous <=3 valid days.
    prev_ref_idx = prev_days_idx[-3:]
    if prev_ref_idx:
        ref_days = [daily[d] for d in prev_ref_idx]
        ref_profile = [median([d[i] for d in ref_days]) for i in range(len(cur))]
        features["anomaly_l1_prev3"] = _distance_to_ref(cur, ref_profile, "l1")
        features["anomaly_l2_prev3"] = _distance_to_ref(cur, ref_profile, "l2")
    else:
        features["anomaly_l1_prev3"] = math.nan
        features["anomaly_l2_prev3"] = math.nan

    # Ratio style features (causal).
    prev_mean3 = _rolling_mean(hist_means[:-1], 3) if len(hist_means) > 1 else math.nan
    features["ratio_mean_vs_prev3"] = (cur_mean / prev_mean3) if math.isfinite(prev_mean3) and prev_mean3 != 0 else math.nan

    return features


def _min_event_distance_days(rat_id: str, d: date, events_by_rat: dict[str, set[date]]) -> int:
    events = events_by_rat.get(rat_id, set())
    if not events:
        return 10**9
    return min(abs((d - e).days) for e in events)


def _match_quiet_controls(
    positives: list[DailySample],
    negatives: list[DailySample],
    target_ratio: float,
) -> set[tuple[str, date]]:
    """Select quiet negatives matched by rat and month where possible."""
    by_key: dict[tuple[str, int], list[DailySample]] = defaultdict(list)
    for n in negatives:
        by_key[(n.rat_id, n.sample_date.month)].append(n)

    selected: set[tuple[str, date]] = set()
    for p in positives:
        key = (p.rat_id, p.sample_date.month)
        pool = by_key.get(key, [])
        for n in pool:
            if len(selected) >= int(math.ceil(target_ratio * len(positives))):
                break
            selected.add((n.rat_id, n.sample_date))
        if len(selected) >= int(math.ceil(target_ratio * len(positives))):
            break

    # Fallback: fill from remaining negatives if strict matching is sparse.
    if len(selected) < int(math.ceil(target_ratio * len(positives))):
        for n in negatives:
            key = (n.rat_id, n.sample_date)
            if key in selected:
                continue
            selected.add(key)
            if len(selected) >= int(math.ceil(target_ratio * len(positives))):
                break
    return selected


def main() -> None:
    args = _parse_args()
    profiles_dir = Path(args.profiles_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_exported_metadata(profiles_dir / "samples_10days_metadata.csv")
    rows = _read_csv_rows(profiles_dir / "samples_10days_nanpad.csv")
    if len(rows) != len(metadata):
        raise ValueError(f"rows mismatch: profile_rows={len(rows)} metadata_rows={len(metadata)}")

    # Build event-date lookup per rat (from before windows).
    events_by_rat: dict[str, set[date]] = defaultdict(set)
    for m in metadata:
        if m["window_direction"] == "before":
            events_by_rat[m["rat_id"]].add(datetime.fromisoformat(m["event_date"]).date())

    candidates: list[tuple[DailySample, dict[str, float]]] = []
    positives: list[DailySample] = []
    negatives_quiet_pool: list[DailySample] = []
    negatives_any_pool: list[DailySample] = []

    for i, (m, raw_row) in enumerate(zip(metadata, rows)):
        if m["window_direction"] != "before":
            continue
        rat_id = m["rat_id"]
        event_date = datetime.fromisoformat(m["event_date"]).date()
        key = (rat_id, m["event_date"], m["window_direction"])
        excluded_days = ARTIFACT_DAYS_BY_KEY.get(key, set())

        norm_row = _normalize_minus1_plus1(raw_row)
        daily = _daily_profiles(norm_row, window_days=int(args.window_days))
        window_dates = [datetime.strptime(x, "%Y_%m_%d").date() for x in m["window_dates"].split(";")]

        for day_idx, day_date in enumerate(window_dates):
            if day_idx in excluded_days:
                continue

            # Binary risk target in next 24h (day granularity).
            label_next24h = int((day_date + timedelta(days=1)) in events_by_rat.get(rat_id, set()))
            sample = DailySample(
                rat_id=rat_id,
                sample_date=day_date,
                source_event_date=event_date,
                day_idx=day_idx,
                label_next24h=label_next24h,
                is_quiet_control=0,
                group_event_date=event_date,
            )
            feats = _build_day_level_features(daily=daily, day_idx=day_idx, excluded_days=excluded_days)
            candidates.append((sample, feats))
            if label_next24h == 1:
                positives.append(sample)
            else:
                negatives_any_pool.append(sample)
                min_dist = _min_event_distance_days(rat_id, day_date, events_by_rat)
                if min_dist > int(args.quiet_buffer_days):
                    negatives_quiet_pool.append(sample)

    control_pool = negatives_quiet_pool if negatives_quiet_pool else negatives_any_pool
    quiet_selected = _match_quiet_controls(
        positives=positives,
        negatives=control_pool,
        target_ratio=float(args.quiet_ratio),
    )

    dataset_rows: list[dict[str, str | float | int]] = []
    for sample, feats in candidates:
        include = sample.label_next24h == 1 or (sample.rat_id, sample.sample_date) in quiet_selected
        if not include:
            continue
        row: dict[str, str | float | int] = {
            "rat_id": sample.rat_id,
            "sample_date": sample.sample_date.isoformat(),
            "source_event_date": sample.source_event_date.isoformat(),
            "group_event_date": sample.group_event_date.isoformat(),
            "day_idx": sample.day_idx,
            "label_next24h": sample.label_next24h,
            "is_quiet_control": int(sample.label_next24h == 0),
        }
        row.update(feats)
        dataset_rows.append(row)

    out_csv = out_dir / "risk_daily_dataset.csv"
    if not dataset_rows:
        raise RuntimeError("No rows generated for risk dataset.")

    fieldnames = list(dataset_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dataset_rows)

    summary = {
        "n_rows": len(dataset_rows),
        "n_positive": sum(int(r["label_next24h"]) for r in dataset_rows),
        "n_negative": sum(1 - int(r["label_next24h"]) for r in dataset_rows),
        "n_quiet_controls": sum(int(r["is_quiet_control"]) for r in dataset_rows),
        "n_quiet_pool_strict": len(negatives_quiet_pool),
        "n_quiet_pool_fallback_any": len(negatives_any_pool),
        "used_fallback_controls": len(negatives_quiet_pool) == 0,
        "quiet_buffer_days": int(args.quiet_buffer_days),
        "quiet_ratio_target": float(args.quiet_ratio),
        "artifact_days_by_key": {str(k): sorted(v) for k, v in ARTIFACT_DAYS_BY_KEY.items()},
        "profiles_dir": str(profiles_dir),
        "output_csv": str(out_csv),
    }
    (out_dir / "risk_daily_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
