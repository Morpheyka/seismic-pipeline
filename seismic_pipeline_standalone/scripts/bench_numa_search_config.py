#!/usr/bin/env python3
"""Synthetic benchmark for worker/BLAS/NUMA configurations."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian.parallel_search import (  # noqa: E402
    ParallelSearchConfig,
    _configure_blas_threads,
    _fit_models_parallel,
    _generate_feature_configs,
    _generate_likelihood_combos,
)

try:  # optional, benchmark still works without psutil
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark NUMA/BLAS worker layouts on synthetic data.")
    p.add_argument("--out-dir", default="./run_output_8day_parallel_full/benchmarks_synth")
    p.add_argument("--events", type=int, default=36, help="Synthetic number of events (rows).")
    p.add_argument("--n-points-per-day", type=int, default=24)
    p.add_argument("--n-days", type=int, default=8)
    p.add_argument("--draws", type=int, default=80)
    p.add_argument("--tune", type=int, default=120)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--n-models", type=int, default=12, help="Number of model configs per layout.")
    p.add_argument("--seed", type=int, default=20260708)
    return p.parse_args()


def _build_synthetic_data(
    *,
    n_events: int,
    n_days: int,
    n_points_per_day: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    width = int(n_days * n_points_per_day)
    x = np.linspace(0.0, 1.0, width, dtype=float)
    data_raw = np.zeros((n_events, width), dtype=float)
    for i in range(n_events):
        cp_day = int(rng.integers(2, n_days - 1))
        cp_idx = cp_day * n_points_per_day
        before = -0.3 + 0.25 * np.sin(2.0 * np.pi * x[:cp_idx]) + rng.normal(0, 0.12, cp_idx)
        after = 0.35 + 0.35 * np.sin(2.0 * np.pi * x[cp_idx:]) + rng.normal(0, 0.12, width - cp_idx)
        arr = np.concatenate([before, after], axis=0)
        data_raw[i] = arr + float(rng.normal(0.0, 0.06))
    # keep bounded range similar to expected profile-like values
    data_raw = np.clip(data_raw, -1.0, 1.0)
    data_norm = data_raw.copy()
    good_indices = np.arange(n_events, dtype=int)
    return data_norm, data_raw, good_indices


def _build_model_configs(
    *,
    n_models: int,
    n_chunks: int,
    n_points_per_day: int,
) -> list[dict[str, Any]]:
    feature_cfgs = _generate_feature_configs(max_features=2, feature_groups=["daily", "day", "night"])
    configs: list[dict[str, Any]] = []
    for feat_cfg in feature_cfgs:
        for param_sel in _generate_likelihood_combos(
            feature_selection=feat_cfg,
            mean_likelihoods=["normal", "student_t"],
            range_likelihoods=["beta", "interval_inflated_beta"],
        ):
            configs.append(
                {
                    "rem_profile_params": {
                        "n_points_per_day": int(n_points_per_day),
                        "overlap": 0.5,
                        "rem_stage": 2,
                    },
                    "n_chunks": int(n_chunks),
                    "feature_selection": feat_cfg,
                    "parameter_selection": param_sel,
                    "tau_threshold": 5.0,
                }
            )
    # deterministic subset
    configs = sorted(configs, key=lambda c: json.dumps(c, sort_keys=True))
    return configs[: max(1, int(n_models))]


def _monitor_runtime(stop_event: threading.Event, sample_interval_sec: float = 0.25) -> dict[str, float]:
    if psutil is None:
        while not stop_event.wait(sample_interval_sec):
            pass
        return {"cpu_util_pct_mean": float("nan"), "peak_rss_gb": float("nan")}
    proc = psutil.Process(os.getpid())
    cpu_samples: list[float] = []
    rss_samples: list[int] = []
    # prime CPU counter
    psutil.cpu_percent(interval=None)
    while not stop_event.wait(sample_interval_sec):
        cpu_samples.append(float(psutil.cpu_percent(interval=None)))
        rss_total = 0
        try:
            rss_total += int(proc.memory_info().rss)
            for ch in proc.children(recursive=True):
                rss_total += int(ch.memory_info().rss)
        except Exception:
            pass
        rss_samples.append(rss_total)
    cpu_mean = float(np.mean(cpu_samples)) if cpu_samples else float("nan")
    peak_rss = float(max(rss_samples) / (1024**3)) if rss_samples else float("nan")
    return {"cpu_util_pct_mean": cpu_mean, "peak_rss_gb": peak_rss}


def _run_layout(
    *,
    name: str,
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    good_indices: np.ndarray,
    model_configs: list[dict[str, Any]],
    n_points_per_day: int,
    base_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = ParallelSearchConfig(**base_cfg)
    _configure_blas_threads(cfg)
    stop = threading.Event()
    monitor_result: dict[str, float] = {}

    def _monitor() -> None:
        monitor_result.update(_monitor_runtime(stop))

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()
    t0 = time.perf_counter()
    results = _fit_models_parallel(
        configs=model_configs,
        data_norm=data_norm,
        data_raw=data_raw,
        good_indices=good_indices,
        n_points=n_points_per_day,
        search_config=cfg,
        verbose=False,
        progress_desc=None,
    )
    elapsed = time.perf_counter() - t0
    stop.set()
    mon.join(timeout=2.0)

    df = pd.DataFrame(results)
    ok_df = df[df.get("status", "") == "ok"].copy()
    n_models = len(model_configs)
    models_per_hour = (3600.0 * n_models / elapsed) if elapsed > 0 else float("nan")
    summary = {
        "layout": name,
        "n_jobs": int(cfg.n_jobs),
        "chains": int(cfg.chains),
        "blas_threads_per_worker": (
            int(cfg.blas_threads_per_worker)
            if cfg.blas_threads_per_worker is not None
            else float("nan")
        ),
        "worker_affinity_mode": str(cfg.worker_affinity_mode),
        "reserve_physical_cores_per_numa_node": int(cfg.reserve_physical_cores_per_numa_node),
        "wall_time_sec": float(elapsed),
        "models_total": int(n_models),
        "models_ok": int(len(ok_df)),
        "models_per_hour": float(models_per_hour),
        "n_divergences_sum": int(pd.to_numeric(df.get("n_divergences"), errors="coerce").fillna(0).sum()),
        "r_hat_max": float(pd.to_numeric(df.get("r_hat_max"), errors="coerce").max()),
        "ess_min_bulk": float(pd.to_numeric(df.get("ess_min_bulk"), errors="coerce").min()),
        "elpd_loo_median": float(pd.to_numeric(df.get("elpd_loo"), errors="coerce").median()),
        "loo_pareto_k_max": float(pd.to_numeric(df.get("loo_pareto_k_max"), errors="coerce").max()),
    }
    summary.update(monitor_result)
    return summary


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_norm, data_raw, good_indices = _build_synthetic_data(
        n_events=int(args.events),
        n_days=int(args.n_days),
        n_points_per_day=int(args.n_points_per_day),
        seed=int(args.seed),
    )
    model_configs = _build_model_configs(
        n_models=int(args.n_models),
        n_chunks=int(args.n_days),
        n_points_per_day=int(args.n_points_per_day),
    )

    common = dict(
        n_points_choices=[int(args.n_points_per_day)],
        overlap_choices=[0.5],
        max_features=2,
        feature_groups=["daily", "day", "night"],
        n_chunks_mode="window_days",
        mean_likelihoods=["normal", "student_t"],
        range_likelihoods=["beta", "interval_inflated_beta"],
        draws=int(args.draws),
        tune=int(args.tune),
        chains=int(args.chains),
        gc_frequency=8,
        nuts_backend="blackjax",
        jax_chain_method="parallel",
        tau_mode="marginalized",
        tau_lower=3,
        tau_upper=int(args.n_days),
        tau_threshold=5.0,
        pareto_threshold=0.7,
        max_pareto_retries=2,
        out_dir=out_dir,
        window_days=int(args.n_days),
    )

    layouts: list[tuple[str, dict[str, Any]]] = [
        (
            "baseline_4w_blas1_no_affinity",
            dict(common, n_jobs=4, cores_per_chain=1, blas_threads_per_worker=1, worker_affinity_mode="none"),
        ),
        (
            "numa_spread_4w_blas1",
            dict(
                common,
                n_jobs=4,
                cores_per_chain=1,
                blas_threads_per_worker=1,
                worker_affinity_mode="numa_spread",
            ),
        ),
        (
            "numa_partition_4w_blas1_reserve3",
            dict(
                common,
                n_jobs=4,
                cores_per_chain=1,
                blas_threads_per_worker=1,
                worker_affinity_mode="numa_partition",
                reserve_physical_cores_per_numa_node=3,
            ),
        ),
        (
            "numa_partition_4w_blas2_reserve3",
            dict(
                common,
                n_jobs=4,
                cores_per_chain=2,
                blas_threads_per_worker=2,
                worker_affinity_mode="numa_partition",
                reserve_physical_cores_per_numa_node=3,
            ),
        ),
        (
            "numa_partition_3w_blas1_reserve3",
            dict(
                common,
                n_jobs=3,
                cores_per_chain=1,
                blas_threads_per_worker=1,
                worker_affinity_mode="numa_partition",
                reserve_physical_cores_per_numa_node=3,
            ),
        ),
        (
            "numa_partition_3w_blas2_reserve3",
            dict(
                common,
                n_jobs=3,
                cores_per_chain=2,
                blas_threads_per_worker=2,
                worker_affinity_mode="numa_partition",
                reserve_physical_cores_per_numa_node=3,
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for idx, (name, cfg_dict) in enumerate(layouts, start=1):
        print(f"[bench] {idx}/{len(layouts)} {name}", flush=True)
        row = _run_layout(
            name=name,
            data_norm=data_norm,
            data_raw=data_raw,
            good_indices=good_indices,
            model_configs=model_configs,
            n_points_per_day=int(args.n_points_per_day),
            base_cfg=cfg_dict,
        )
        rows.append(row)
        print(
            f"[bench] done {name}: wall={row['wall_time_sec']:.1f}s "
            f"models/hr={row['models_per_hour']:.1f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("models_per_hour", ascending=False).reset_index(drop=True)
    out_csv = out_dir / "synthetic_numa_benchmark.csv"
    df.to_csv(out_csv, index=False)
    top = df.iloc[0].to_dict() if not df.empty else {}
    out_json = out_dir / "synthetic_numa_benchmark_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "benchmark_args": vars(args),
                "winner_by_models_per_hour": top,
                "all_rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=lambda x: x if isinstance(x, (int, float, str, bool, type(None))) else str(x),
        ),
        encoding="utf-8",
    )
    print(f"[bench] wrote {out_csv}")
    print(f"[bench] wrote {out_json}")


if __name__ == "__main__":
    main()
