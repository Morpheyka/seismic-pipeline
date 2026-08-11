#!/usr/bin/env python3
"""Sweep n_jobs / OMP for blackjax parallel search workload."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from joblib import Parallel, delayed

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    expected_fixed_n_chunk_count,
    prepare_model_data,
)

CSV = PROJECT_ROOT.parent / (
    "run_output_8day_smoke_test/profile_cache/rem_n24_ov0.50_stage2/samples_10days_nanpad.csv"
)
DRAWS = 800
TUNE = 800
CHAINS = 4

_PREP = prepare_model_data(csv_path=str(CSV))
_DATA_NORM = np.asarray(_PREP["data_norm"])
_DATA_RAW = np.asarray(_PREP["data_raw"])
_N_CHUNKS = expected_fixed_n_chunk_count(n_points_per_day=24, n_days=8)


def _set_omp(omp: int) -> None:
    val = str(omp)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = val


def _fit_one(
    feature_selection: dict,
    parameter_selection: dict,
    omp: int,
    data_norm: np.ndarray,
    data_raw: np.ndarray,
) -> float:
    _set_omp(omp)
    group_data = build_group_data(
        data_norm,
        n_chunks=_N_CHUNKS,
        feature_selection=feature_selection,
        data_raw=data_raw,
        window_days=8,
        n_points_per_day=24,
    )
    model = build_changepoint_model(
        group_data,
        tau_lower=3,
        tau_upper=8,
        parameter_selection=parameter_selection,
        tau_mode="marginalized",
    )
    t0 = time.perf_counter()
    sample_model(
        model,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        nuts_backend="blackjax",
        jax_chain_method="parallel",
        progressbar=False,
    )
    return time.perf_counter() - t0


def _run_batch(n_jobs: int, omp: int) -> float:
    _set_omp(omp)
    configs = [
        ({"concat": ["mean"]}, {"mean": {"likelihood": "student_t"}}),
        (
            {"concat": ["mean", "range"]},
            {"mean": {"likelihood": "student_t"}, "range": {"likelihood": "beta"}},
        ),
    ]
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one)(fs, ps, omp, _DATA_NORM, _DATA_RAW) for fs, ps in configs
    )
    return time.perf_counter() - t0


def main() -> None:
    # Remove any polluted XLA flags from prior runs.
    os.environ.pop("XLA_FLAGS", None)

    print(f"CPU count: {os.cpu_count()}")
    print(f"Backend: blackjax parallel | draws={DRAWS} tune={TUNE} chains={CHAINS}")
    print(f"{'n_jobs':>6} {'omp':>4} {'wall_s':>8} {'per_model':>10}")
    print("-" * 34)

    results: list[tuple[float, int, int]] = []
    for n_jobs in [1, 2, 4, 8, 12, 16, 18]:
        for omp in [1, 2, 4, 8]:
            budget = n_jobs * omp
            if budget > 72:
                continue
            try:
                wall = _run_batch(n_jobs, omp)
                per = wall / 2
                results.append((wall, n_jobs, omp))
                print(f"{n_jobs:6d} {omp:4d} {wall:8.1f} {per:10.1f}")
            except Exception as exc:
                print(f"{n_jobs:6d} {omp:4d}     FAIL {type(exc).__name__}: {str(exc)[:50]}")

    if not results:
        return
    best = min(results, key=lambda r: r[0])
    print("-" * 34)
    print(f"BEST wall: n_jobs={best[1]} omp={best[2]} -> {best[0]:.1f}s for 2 models")


if __name__ == "__main__":
    main()
