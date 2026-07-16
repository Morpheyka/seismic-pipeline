#!/usr/bin/env python3
"""Full exhaustive parallel search on 8-day REM events."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.parallel_search import ParallelSearchConfig, run_parallel_search
from seismic_pipeline.config import FULL_EXHAUSTIVE_EVENTS_8DAY, default_export_base_cfg
from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
from seismic_pipeline.mod.threading_config import configure_threading
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full 8-day exhaustive search.")
    parser.add_argument("--out-dir", default="./run_output_8day_parallel_full", help="Output directory.")
    parser.add_argument("--resume-from-csv", default=None, help="Optional CSV path for resume mode.")
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Run only smoke validation and statistics.",
    )
    return parser.parse_args()


def _distribution_stats(values: np.ndarray, *, near_tol: float = 0.05) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in (
            "n",
            "min",
            "max",
            "mean",
            "std",
            "median",
            "iqr",
            "skewness",
            "kurtosis",
            "frac_near_0",
            "frac_near_1",
            "frac_negative",
            "frac_positive",
            "outlier_rate_iqr",
        )}
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    mu = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std > 0 and arr.size > 2:
        z = (arr - mu) / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    out_rate = float(np.mean((arr < lo) | (arr > hi))) if iqr > 0 else 0.0
    return {
        "n": float(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": mu,
        "std": std,
        "median": float(np.median(arr)),
        "iqr": float(iqr),
        "skewness": skew,
        "kurtosis": kurt,
        "frac_near_0": float(np.mean(np.abs(arr - 0.0) <= near_tol)),
        "frac_near_1": float(np.mean(np.abs(arr - 1.0) <= near_tol)),
        "frac_negative": float(np.mean(arr < 0.0)),
        "frac_positive": float(np.mean(arr > 0.0)),
        "outlier_rate_iqr": out_rate,
    }


def _recommend_mean_likelihood(mean_stats: pd.DataFrame) -> tuple[list[str], str]:
    heavy_tail = False
    if not mean_stats.empty:
        heavy_tail = bool(
            (mean_stats["outlier_rate_iqr"].fillna(0.0) > 0.05).any()
            or (mean_stats["kurtosis"].abs().fillna(0.0) > 3.0).any()
            or (mean_stats["skewness"].abs().fillna(0.0) > 1.0).any()
        )
    if heavy_tail:
        return ["student_t", "normal"], "student_t preferred: heavy tails/outliers detected"
    return ["normal", "student_t"], "normal preferred: near-symmetric, weak outliers"


def _compute_metric_stats(
    *,
    out_dir: Path,
    export_base_cfg: dict[str, Any],
    n_points_per_day: int = 24,
    overlap: float = 0.5,
    rem_stage: int = 2,
    window_days: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_export_dir = out_dir / "metric_stats_profile_cache"
    metric_export_dir.mkdir(parents=True, exist_ok=True)
    export_cfg = dict(export_base_cfg)
    export_cfg.update(
        {
            "output_dir": str(metric_export_dir),
            "window_days": int(window_days),
            "n_points_per_day": int(n_points_per_day),
            "overlap": float(overlap),
            "rem_stage": int(rem_stage),
        }
    )
    export_result = export_rem_profiles_10days_cached_only(**export_cfg)
    prep = prepare_model_data(csv_path=export_result["paths"]["nanpad_output_csv"])
    data_norm = np.asarray(prep["data_norm"], dtype=float)
    data_raw = np.asarray(prep["data_raw"], dtype=float)
    group_data = build_group_data(
        data_norm,
        n_chunks=int(window_days),
        feature_selection={"daily": ["mean", "range"], "day": ["mean", "range"], "night": ["mean", "range"]},
        data_raw=data_raw,
        window_days=int(window_days),
        n_points_per_day=int(n_points_per_day),
    )

    rows: list[dict[str, Any]] = []
    for group_name, metrics in group_data.items():
        for metric_name, df in metrics.items():
            metric_id = f"{group_name}_{metric_name}"
            vals = df.to_numpy(dtype=float).reshape(-1)
            stats = _distribution_stats(vals)
            row = {"metric_id": metric_id, "group": group_name, "metric": metric_name}
            row.update(stats)
            if metric_name == "range":
                scaled = vals / 2.0
                scaled_stats = _distribution_stats(scaled)
                row["scaled_frac_near_1"] = scaled_stats["frac_near_1"]
                row["scaled_frac_near_0"] = scaled_stats["frac_near_0"]
                row["scaled_max"] = scaled_stats["max"]
            else:
                row["scaled_frac_near_1"] = float("nan")
                row["scaled_frac_near_0"] = float("nan")
                row["scaled_max"] = float("nan")
            rows.append(row)

    stats_df = pd.DataFrame(rows).sort_values(["group", "metric"]).reset_index(drop=True)
    mean_stats = stats_df[stats_df["metric"] == "mean"].copy()
    mean_likes, mean_reason = _recommend_mean_likelihood(mean_stats)
    recommendation = {
        "mean_likelihood_candidates": mean_likes,
        "range_likelihood_candidates": ["beta", "interval_inflated_beta"],
        "range_support_upper": 2.0,
        "range_scaling_for_beta": "range_scaled = range / 2.0",
        "notes": {
            "mean": mean_reason,
            "range": "range metrics are bounded in [0,2], use beta-family on range/2 in [0,1]",
        },
        "metrics_covered": sorted(stats_df["metric_id"].tolist()),
    }
    return stats_df, recommendation


def _warm_blackjax_backend() -> None:
    """Compile/warm a tiny blackjax model once before heavy parallel search."""
    tiny = np.linspace(0.0, 1.0, 8 * 24, dtype=float).reshape(1, -1)
    group_data = build_group_data(
        tiny,
        n_chunks=8,
        feature_selection={"daily": ["mean"]},
        data_raw=tiny,
        window_days=8,
        n_points_per_day=24,
    )
    model = build_changepoint_model(
        group_data,
        tau_lower=2,
        tau_upper=8,
        parameter_selection={"mean": {"likelihood": "normal"}},
        tau_mode="marginalized",
    )
    _ = sample_model(
        model,
        draws=10,
        tune=10,
        nuts_backend="blackjax",
        chains=1,
        progressbar=False,
        jax_chain_method="parallel",
        jax_var_names=["changepoint_pointwise_log_lik", "tau_probs", "tau_support"],
        materialize_posterior_vars=["changepoint_pointwise_log_lik", "tau_probs", "tau_support"],
    )


def _run_smoke(
    *,
    out_dir: Path,
    export_base_cfg: dict[str, Any],
    mean_likelihoods: list[str],
) -> pd.DataFrame:
    smoke_cfg = ParallelSearchConfig(
        n_points_choices=[24],
        overlap_choices=[0.5],
        max_features=2,
        feature_groups=["daily", "day", "night"],
        n_chunks_mode="window_days",
        mean_likelihoods=mean_likelihoods,
        range_likelihoods=["beta", "interval_inflated_beta"],
        draws=100,
        tune=200,
        chains=2,
        cores_per_chain=2,
        blas_threads_per_worker=2,
        n_jobs=1,
        nuts_backend="blackjax",
        tau_mode="marginalized",
        tau_lower=3,
        tau_upper=8,
        max_pareto_retries=2,
        worker_affinity_mode="numa_partition",
        reserve_physical_cores_per_numa_node=3,
        out_dir=out_dir / "smoke",
    )
    return run_parallel_search(
        config=smoke_cfg,
        export_base_cfg=export_base_cfg,
        verbose=True,
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep BLAS pressure bounded for nested joblib + chain workers.
    configure_threading(cores=16, threads_per_job=max(1, 10 // 4))

    export_base_cfg = default_export_base_cfg(output_dir=str(out_dir / "profile_cache"))
    export_base_cfg["events"] = [dict(x) for x in FULL_EXHAUSTIVE_EVENTS_8DAY]
    export_base_cfg["window_days"] = 8
    export_base_cfg["drop_incomplete_events"] = True

    stats_df, recommendation = _compute_metric_stats(
        out_dir=out_dir,
        export_base_cfg=export_base_cfg,
        n_points_per_day=24,
        overlap=0.5,
        rem_stage=2,
        window_days=8,
    )
    stats_csv = out_dir / "metric_distribution_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    rec_json = out_dir / "likelihood_recommendation.json"
    rec_json.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stats] wrote {stats_csv}")
    print(f"[stats] wrote {rec_json}")

    _warm_blackjax_backend()
    print("[warmup] blackjax warmup complete")

    _ = _run_smoke(
        out_dir=out_dir,
        export_base_cfg=export_base_cfg,
        mean_likelihoods=list(recommendation["mean_likelihood_candidates"]),
    )
    print("[smoke] completed")

    if args.skip_full:
        print("[full] skipped by --skip-full")
        return

    full_cfg = ParallelSearchConfig(
        n_points_choices=[12, 24, 48],
        overlap_choices=[0.0, 0.25, 0.5],
        max_features=3,
        feature_groups=["daily", "day", "night"],
        n_chunks_mode="window_days",
        mean_likelihoods=list(recommendation["mean_likelihood_candidates"]),
        range_likelihoods=["beta", "interval_inflated_beta"],
        draws=2000,
        tune=4000,
        chains=4,
        cores_per_chain=2,
        blas_threads_per_worker=2,
        n_jobs=3,
        gc_frequency=10,
        nuts_backend="blackjax",
        jax_chain_method="parallel",
        tau_mode="marginalized",
        tau_lower=3,
        tau_upper=8,
        tau_threshold=5.0,
        pareto_threshold=0.7,
        max_pareto_retries=2,
        worker_affinity_mode="numa_partition",
        reserve_physical_cores_per_numa_node=3,
        out_dir=out_dir,
    )
    results = run_parallel_search(
        config=full_cfg,
        export_base_cfg=export_base_cfg,
        resume_from_csv=Path(args.resume_from_csv) if args.resume_from_csv else None,
        verbose=True,
    )
    print(f"[full] done. rows={len(results)}")


if __name__ == "__main__":
    main()
