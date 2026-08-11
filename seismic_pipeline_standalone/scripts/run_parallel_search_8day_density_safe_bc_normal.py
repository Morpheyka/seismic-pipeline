#!/usr/bin/env python3
"""Density-safe 8-day parallel search — range = beta_constrained + normal.

After ZOIB/IIB piled E[τ] at the prior floor (τ=2), range families are frozen to:
  - beta_constrained (α,β ≥ 1, no U-shape)
  - normal (same Gaussian family that behaved well for mean)

Grid otherwise unchanged (rev. 3):
  N ∈ {12, 24}, overlap ∈ {0, 0.25, 0.5}
  groups: daily | day | night  × mean/range (≤3 blocks)
  mean ∈ {student_t, skew_normal}
  τ ∈ {2…8}, day-mask ON, K=6

Usage (from seismic_pipeline_standalone/):
  python scripts/run_parallel_search_8day_density_safe_bc_normal.py --smoke-only \\
      --out-dir ./run_output_8day_density_safe_bc_normal
  python scripts/run_parallel_search_8day_density_safe_bc_normal.py --skip-smoke \\
      --out-dir ./run_output_8day_density_safe_bc_normal --draws 500 --tune 1000 --chains 2
"""
from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="./run_output_8day_density_safe_bc_normal",
        help="Output directory for search + profile_cache.",
    )
    p.add_argument("--resume-from-csv", default=None, help="Resume CSV path.")
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run 3 smoke configs only (no full grid).",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke and go straight to full/exploratory search.",
    )
    p.add_argument(
        "--include-plain-beta-diag",
        action="store_true",
        help="Also fit plain beta (rank_eligible=False via DIAG gate).",
    )
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=4000)
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--nuts-backend", default="blackjax")
    return p.parse_args()


def _export_base(out_dir: Path) -> dict[str, Any]:
    cfg = default_export_base_cfg(output_dir=str(out_dir / "profile_cache"))
    cfg["events"] = [dict(x) for x in FULL_EXHAUSTIVE_EVENTS_8DAY]
    cfg["window_days"] = 8
    cfg["drop_incomplete_events"] = False
    return cfg


def _range_likes(*, include_plain_beta: bool) -> list[str]:
    likes = ["beta_constrained", "normal"]
    if include_plain_beta:
        likes = ["beta", *likes]
    return likes


def _smoke_configs() -> list[dict[str, Any]]:
    """Three representative configs: BC primary, normal range, mean+range normal."""
    return [
        {
            "tag": "smoke_primary_a_bc",
            "feature_selection": {"daily": ["mean", "range"]},
            "parameter_selection": {
                "mean": {"likelihood": "student_t"},
                "range": {"likelihood": "beta_constrained", "support_upper": 2.0},
            },
        },
        {
            "tag": "smoke_range_normal",
            "feature_selection": {"daily": ["range"]},
            "parameter_selection": {
                "range": {"likelihood": "normal"},
            },
        },
        {
            "tag": "smoke_mean_range_normal",
            "feature_selection": {"daily": ["mean", "range"]},
            "parameter_selection": {
                "mean": {"likelihood": "student_t"},
                "range": {"likelihood": "normal"},
            },
        },
    ]


def _run_smoke(out_dir: Path, export_base_cfg: dict[str, Any]) -> pd.DataFrame:
    """Fit three short MCMC configs with mask ON on N=24 ov=0.5."""
    cache_dir = out_dir / "smoke" / "profile_cache" / "rem_n24_ov0.50_stage2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    export_cfg = dict(export_base_cfg)
    export_cfg.update(
        {
            "output_dir": str(cache_dir),
            "n_points_per_day": 24,
            "overlap": 0.5,
            "rem_stage": 2,
            "window_days": 8,
            "drop_incomplete_events": False,
        }
    )
    export_result = export_rem_profiles_10days_cached_only(**export_cfg)
    csv_path = export_result["paths"]["nanpad_output_csv"]
    prep = prepare_model_data(
        csv_path=csv_path,
        day_mask=True,
        apply_artifacts=True,
        min_valid_days=6,
        n_points_per_day=24,
        window_days=8,
    )
    data_raw = np.asarray(prep["data_raw"], dtype=float)
    data_norm = np.asarray(prep["data_norm"], dtype=float)
    day_valid = prep.get("day_valid")
    print(
        f"[smoke] n_events={data_raw.shape[0]} n_masked_days={prep.get('n_masked_days', 0)} "
        f"csv={csv_path}"
    )

    rows: list[dict[str, Any]] = []
    for cell in _smoke_configs():
        tag = cell["tag"]
        print(f"[smoke] fitting {tag} …", flush=True)
        group_data = build_group_data(
            data_norm,
            n_chunks=8,
            feature_selection=cell["feature_selection"],
            data_raw=data_raw,
            window_days=8,
            n_points_per_day=24,
            day_valid=day_valid,
        )
        model = build_changepoint_model(
            group_data,
            tau_lower=2,
            tau_upper=8,
            parameter_selection=cell["parameter_selection"],
            tau_mode="marginalized",
        )
        trace = sample_model(
            model,
            draws=80,
            tune=120,
            nuts_backend="blackjax",
            chains=2,
            progressbar=False,
            jax_chain_method="parallel",
            jax_var_names=["changepoint_pointwise_log_lik", "tau_probs", "tau_support", "tau_mean"],
            materialize_posterior_vars=[
                "changepoint_pointwise_log_lik",
                "tau_probs",
                "tau_support",
                "tau_mean",
            ],
        )
        tau_mean = float(np.asarray(trace.posterior["tau_mean"]).mean())
        row = {
            "tag": tag,
            "status": "ok",
            "n_events": int(data_raw.shape[0]),
            "n_masked_days": int(prep.get("n_masked_days", 0) or 0),
            "tau_mean": tau_mean,
            "feature_selection": json.dumps(cell["feature_selection"]),
            "parameter_selection": json.dumps(cell["parameter_selection"]),
        }
        rows.append(row)
        print(f"[smoke] {tag} tau_mean={tau_mean:.3f}", flush=True)
        del trace, model, group_data

    df = pd.DataFrame(rows)
    smoke_dir = out_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    out_csv = smoke_dir / "smoke_density_safe.csv"
    df.to_csv(out_csv, index=False)
    print(f"[smoke] wrote {out_csv}")
    return df


def _full_config(args: argparse.Namespace, out_dir: Path) -> ParallelSearchConfig:
    return ParallelSearchConfig(
        n_points_choices=[12, 24],
        overlap_choices=[0.0, 0.25, 0.5],
        max_features=3,
        feature_groups=["daily", "day", "night"],
        n_chunks_mode="window_days",
        mean_likelihoods=["student_t", "skew_normal"],
        range_likelihoods=_range_likes(include_plain_beta=bool(args.include_plain_beta_diag)),
        draws=int(args.draws),
        tune=int(args.tune),
        chains=int(args.chains),
        cores_per_chain=2,
        blas_threads_per_worker=2,
        n_jobs=int(args.n_jobs),
        gc_frequency=10,
        nuts_backend=str(args.nuts_backend),
        jax_chain_method="parallel",
        tau_mode="marginalized",
        tau_lower=2,
        tau_upper=8,
        tau_threshold=5.0,
        pareto_threshold=0.7,
        max_pareto_retries=0,
        record_pareto_events=True,
        worker_affinity_mode="numa_partition",
        reserve_physical_cores_per_numa_node=3,
        day_mask=True,
        day_mask_apply_artifacts=True,
        min_valid_days=6,
        plain_beta_diagnostic_only=True,
        window_days=8,
        rem_stage=2,
        out_dir=out_dir,
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_threading(cores=16, threads_per_job=max(1, 10 // 4))

    export_base_cfg = _export_base(out_dir)
    meta = {
        "protocol": "density_safe_range_bc_normal",
        "n_points": [12, 24],
        "groups": ["daily", "day", "night"],
        "mean_likelihoods": ["student_t", "skew_normal"],
        "range_likelihoods": _range_likes(include_plain_beta=bool(args.include_plain_beta_diag)),
        "tau": [2, 8],
        "day_mask": True,
        "min_valid_days": 6,
        "drop_incomplete_events": False,
        "max_pareto_retries": 0,
        "note": "ZOIB/IIB dropped after E[tau] floor artifact; range=beta_constrained|normal",
    }
    (out_dir / "density_safe_protocol.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.skip_smoke:
        smoke_df = _run_smoke(out_dir, export_base_cfg)
        if smoke_df.empty or not (smoke_df["status"] == "ok").all():
            raise SystemExit("[smoke] FAILED — aborting full search")
        print("[smoke] OK")

    if args.smoke_only:
        print("[full] skipped (--smoke-only)")
        return

    cfg = _full_config(args, out_dir)
    results = run_parallel_search(
        config=cfg,
        export_base_cfg=export_base_cfg,
        resume_from_csv=Path(args.resume_from_csv) if args.resume_from_csv else None,
        verbose=True,
    )
    print(f"[full] done. rows={len(results)}")
    print(f"[full] csv={out_dir / 'exhaustive_search_parallel.csv'}")
    print(
        f"[full] resume: python scripts/run_parallel_search_8day_density_safe_bc_normal.py "
        f"--skip-smoke --out-dir {out_dir} "
        f"--resume-from-csv {out_dir / 'exhaustive_search_parallel.checkpoint.csv'}"
    )


if __name__ == "__main__":
    main()
