#!/usr/bin/env python3
"""Launch parallel exhaustive search with profile-level cache."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.config import default_export_base_cfg
from seismic_pipeline.mod.threading_config import configure_threading

# Configure threading before importing numpy/pymc-heavy modules.
configure_threading(cores=16, threads_per_job=max(1, 10 // 4))

from seismic_pipeline.bayesian.parallel_search import (  # noqa: E402
    ParallelSearchConfig,
    run_parallel_search,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel exhaustive changepoint search.")
    parser.add_argument(
        "--out-dir",
        default="./run_output_8day_parallel",
        help="Output directory for search artifacts.",
    )
    parser.add_argument(
        "--resume-from-csv",
        default=None,
        help="Optional path to an existing CSV with completed fingerprints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    config = ParallelSearchConfig(
        n_points_choices=[12, 24, 48],
        overlap_choices=[0.0, 0.25, 0.5],
        max_features=3,
        draws=500,
        tune=1000,
        chains=4,
        n_jobs=4,
        gc_frequency=10,
        out_dir=out_dir,
    )

    export_base_cfg = default_export_base_cfg(output_dir=str(out_dir / "profile_cache"))
    results = run_parallel_search(
        config=config,
        export_base_cfg=export_base_cfg,
        resume_from_csv=Path(args.resume_from_csv) if args.resume_from_csv else None,
        verbose=True,
    )
    print(f"\nDone. {len(results)} models recorded.")


if __name__ == "__main__":
    main()
