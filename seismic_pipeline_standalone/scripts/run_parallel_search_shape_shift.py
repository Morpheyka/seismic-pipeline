#!/usr/bin/env python3
"""Launch parallel exhaustive search using shape_shift (L1) features only."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force PyTensor to use scipy-openblas before any other imports
_BLAS_LIB_DIR = "/opt/_internal/cpython-3.12.12/lib/python3.12/site-packages/scipy_openblas64/lib"
if os.path.isdir(_BLAS_LIB_DIR):
    import pytensor

    pytensor.config.blas__ldflags = f"-L{_BLAS_LIB_DIR} -lopenblas"
    # Also set environment variables for any subprocess.
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    if _BLAS_LIB_DIR not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{_BLAS_LIB_DIR}:{os.environ['LD_LIBRARY_PATH']}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.mod.threading_config import configure_threading  # noqa: E402

# Configure threading before importing numpy/pymc-heavy modules.
configure_threading(cores=16, threads_per_job=max(1, 10 // 4))

from seismic_pipeline.bayesian.parallel_search import ParallelSearchConfig, run_parallel_search  # noqa: E402
from seismic_pipeline.config import default_export_base_cfg  # noqa: E402

CUSTOM_EVENTS = [
    {"rat_id": "R2", "date": "2022-11-07", "direction": "before"},
    {"rat_id": "R2", "date": "2022-11-18", "direction": "before"},
    {"rat_id": "R2", "date": "2023-04-03", "direction": "before"},
    {"rat_id": "R2", "date": "2023-04-18", "direction": "before"},
    {"rat_id": "R2", "date": "2023-05-03", "direction": "before"},
    {"rat_id": "R2", "date": "2024-09-30", "direction": "before"},
    {"rat_id": "R2", "date": "2024-10-29", "direction": "before"},
    {"rat_id": "R3", "date": "2025-01-23", "direction": "before"},
    {"rat_id": "R3", "date": "2025-03-14", "direction": "before"},
    {"rat_id": "R1", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R2", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R3", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R4", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R1", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R2", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R3", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R4", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R2", "date": "2022-11-07", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2022-11-18", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2023-04-03", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2023-04-11", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2023-04-21", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2024-09-30", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2024-10-29", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-01-23", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-03-14", "direction": "after_reversed"},
    {"rat_id": "R1", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R4", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R1", "date": "2025-07-20", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2025-07-20", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-07-20", "direction": "after_reversed"},
    {"rat_id": "R4", "date": "2025-07-20", "direction": "after_reversed"},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel shape_shift-only changepoint search.")
    parser.add_argument(
        "--out-dir",
        default="./run_output_8day_parallel_shape_shift",
        help="Output directory for search artifacts.",
    )
    parser.add_argument(
        "--resume-from-csv",
        default=None,
        help="Optional path to an existing CSV with completed fingerprints.",
    )
    parser.add_argument(
        "--nuts-backend",
        default="blackjax",
        choices=["pymc", "numpyro", "blackjax"],
        help="NUTS sampler backend (default: blackjax for JAX speed).",
    )
    parser.add_argument(
        "--jax-chain-method",
        default="auto",
        choices=["auto", "parallel", "vectorized"],
        help="NumPyro chain layout when using a JAX backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    config = ParallelSearchConfig(
        n_points_choices=[12, 24, 48],
        overlap_choices=[0.0, 0.25, 0.5],
        max_features=3,
        feature_groups=["concat", "even", "odd"],
        feature_metrics=["shape_shift"],
        n_chunks_mode="shape_shift",
        shape_shift_likelihoods=["lognormal", "gamma"],
        draws=500,
        tune=1000,
        chains=4,
        n_jobs=4,
        gc_frequency=10,
        window_days=8,
        tau_lower=3,
        tau_upper=7,
        nuts_backend=args.nuts_backend,
        jax_chain_method=args.jax_chain_method,
        out_dir=out_dir,
    )

    export_base_cfg = default_export_base_cfg(output_dir=str(out_dir / "profile_cache"))
    export_base_cfg["events"] = CUSTOM_EVENTS
    export_base_cfg["window_days"] = 8

    results = run_parallel_search(
        config=config,
        export_base_cfg=export_base_cfg,
        resume_from_csv=Path(args.resume_from_csv) if args.resume_from_csv else None,
        verbose=True,
    )
    print(f"\nDone. {len(results)} models recorded.")


if __name__ == "__main__":
    main()
