#!/usr/bin/env python3
"""Minimal smoke test for parallel exhaustive changepoint search on a local machine.

Intended for P-core-only thread layout:
  4 MCMC chains (2 models x 2 chains) + 2 BLAS threads = 6 P-core threads.

Usage:
  python scripts/smoke_test_parallel_search.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# Thread control must be set before NumPy / PyTensor / BLAS imports.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian import parallel_search as ps
from seismic_pipeline.bayesian.parallel_search import (  # noqa: E402
    ParallelSearchConfig,
    run_parallel_search,
)
from seismic_pipeline.config.changepoint_defaults import (  # noqa: E402
    default_export_base_cfg,
    validate_rem_profile_params,
)

SMOKE_FEATURE_CONFIGS: list[dict[str, list[str]]] = [
    {"concat": ["mean"]},
    {"concat": ["mean", "range"]},
]

_REQUIRED_CSV_COLUMNS = (
    "tau_q1",
    "tau_q2",
    "tau_q3",
    "tau_hdi_60_lower",
    "tau_hdi_60_upper",
    "tau_hdi_60_width",
)


def _smoke_feature_configs(max_features: int = 2) -> list[dict[str, list[str]]]:
    """Return the two fixed feature configs used in the smoke grid."""
    del max_features
    return [dict(cfg) for cfg in SMOKE_FEATURE_CONFIGS]


def _zipped_rem_profile_grid(self: ParallelSearchConfig) -> list[dict[str, int | float]]:
    """Pair n_points/overlap by index when lengths match (2 profiles, not 2x2 grid)."""
    n_choices = self.n_points_choices
    o_choices = self.overlap_choices
    if len(n_choices) == len(o_choices):
        pairs = zip(n_choices, o_choices, strict=True)
    else:
        pairs = ((n, o) for n in n_choices for o in o_choices)
    out: list[dict[str, int | float]] = []
    for n_points, overlap in pairs:
        out.append(
            validate_rem_profile_params(
                {
                    "n_points_per_day": int(n_points),
                    "overlap": float(overlap),
                    "rem_stage": int(self.rem_stage),
                }
            )
        )
    return out


def _warn_missing_events(export_base_cfg: dict) -> None:
    """Warn when local data is unavailable; export drops incomplete events later."""
    events = export_base_cfg.get("events") or []
    if not events:
        warnings.warn("No events configured in export_base_cfg.", stacklevel=2)
        return

    data_root = Path(str(export_base_cfg.get("local_data_root", "")))
    if not data_root.is_dir():
        warnings.warn(
            f"Local data root not found: {data_root}. "
            "Events with missing hypnograms will be dropped (drop_incomplete_events=True).",
            stacklevel=2,
        )


def _warn_from_export_summaries(out_dir: Path, n_events_requested: int) -> None:
    """Print warnings when profile exports kept fewer events than requested."""
    cache_root = out_dir / "profile_cache"
    if not cache_root.is_dir():
        return
    for summary_path in sorted(cache_root.glob("*/samples_10days_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        exported = int(summary.get("events_exported", n_events_requested))
        total = int(summary.get("events_total", n_events_requested))
        missing_pairs = int(summary.get("missing_pairs_final", 0))
        if exported < n_events_requested or missing_pairs > 0:
            warnings.warn(
                f"{summary_path.parent.name}: exported {exported}/{n_events_requested} events "
                f"({missing_pairs} missing hypnogram day-pairs). Continuing with available data.",
                stacklevel=2,
            )
        if exported < total:
            warnings.warn(
                f"{summary_path.parent.name}: dropped {total - exported} incomplete event(s).",
                stacklevel=2,
            )


def _print_top_models(df, n: int = 3) -> None:
    sort_col = "elpd_loo_per_feature_event"
    if sort_col not in df.columns or df.empty:
        print(f"No results to rank by {sort_col!r}.")
        return
    ranked = df.sort_values(sort_col, ascending=False, na_position="last").head(n)
    print(f"\nTop-{n} models by {sort_col}:")
    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        print(
            f"  {i}. elpd_loo_per_feature_event={row.get(sort_col, float('nan')):.4f}  "
            f"e_tau={row.get('e_tau', float('nan')):.3f}  "
            f"tau_q2={row.get('tau_q2', float('nan')):.3f}  "
            f"tau_hdi_60_width={row.get('tau_hdi_60_width', float('nan')):.3f}  "
            f"features={row.get('features', '')!r}"
        )


def _verify_results(df, csv_path: Path, expected_models: int) -> tuple[list[str], list[str]]:
    """Return (hard failures, soft warnings). r_hat is warning-only for this short run."""
    issues: list[str] = []
    warnings_out: list[str] = []
    if df.empty:
        issues.append("Results dataframe is empty.")
        return issues, warnings_out

    ok_mask = df.get("status", "") == "ok"
    n_ok = int(ok_mask.sum())
    if n_ok != expected_models:
        issues.append(f"Expected {expected_models} successful models, got {n_ok}.")

    failed = df.loc[~ok_mask]
    if not failed.empty:
        for _, row in failed.iterrows():
            issues.append(
                f"Model failed: fingerprint={row.get('fingerprint')} error={row.get('error')}"
            )

    if not csv_path.is_file():
        issues.append(f"Output CSV not found: {csv_path}")
    else:
        header = list(df.columns)
        missing_cols = [c for c in _REQUIRED_CSV_COLUMNS if c not in header]
        if missing_cols:
            issues.append(f"CSV missing columns: {missing_cols}")

    if n_ok > 0 and "r_hat_max" in df.columns:
        ok_df = df.loc[ok_mask]
        rhat = ok_df["r_hat_max"].astype(float)
        bad = ok_df.loc[rhat > 1.05]
        if not bad.empty:
            warnings_out.append(
                f"{len(bad)} model(s) have r_hat_max > 1.05 (max={rhat.max():.3f}); "
                "expected with draws=100/tune=100 — re-check on cluster with full MCMC."
            )

    return issues, warnings_out


def main() -> int:
    ps._generate_feature_configs = _smoke_feature_configs  # type: ignore[attr-defined]
    ParallelSearchConfig.rem_profile_grid = property(_zipped_rem_profile_grid)

    out_dir = Path("./run_output_8day_smoke_test")
    config = ParallelSearchConfig(
        n_points_choices=[24, 12],
        overlap_choices=[0.5, 0.0],
        max_features=2,
        mean_likelihoods=["student_t"],
        range_likelihoods=["beta"],
        draws=100,
        tune=100,
        chains=2,
        cores_per_chain=1,
        tau_mode="marginalized",
        n_jobs=2,
        gc_frequency=2,
        blas_total_cores=4,
        record_pareto_events=True,
        out_dir=out_dir,
    )

    export_base_cfg = default_export_base_cfg(output_dir=str(out_dir))
    n_events_requested = len(export_base_cfg.get("events") or [])
    _warn_missing_events(export_base_cfg)

    expected_models = len(config.rem_profile_grid) * len(SMOKE_FEATURE_CONFIGS)

    print("=== Parallel search smoke test ===")
    print(f"Profiles: {config.rem_profile_grid}")
    print(f"Feature configs: {SMOKE_FEATURE_CONFIGS}")
    print(f"Likelihoods: mean=student_t, range=beta")
    print(f"MCMC: draws={config.draws}, tune={config.tune}, chains={config.chains}")
    print(f"Parallel: n_jobs={config.n_jobs}, gc_frequency={config.gc_frequency}")
    print(f"Expected models: {expected_models}")
    print(f"Output dir: {out_dir.resolve()}")

    t0 = time.perf_counter()
    try:
        results_df = run_parallel_search(
            config=config,
            export_base_cfg=export_base_cfg,
            resume_from_csv=None,
            verbose=True,
        )
    except ValueError as exc:
        warnings.warn(f"Search could not complete: {exc}", stacklevel=2)
        print(f"\nSmoke test aborted: {exc}")
        return 1
    elapsed = time.perf_counter() - t0

    _warn_from_export_summaries(out_dir, n_events_requested)

    csv_path = out_dir / "exhaustive_search_parallel.csv"
    n_fitted = int((results_df.get("status") == "ok").sum()) if not results_df.empty else 0

    print("\n=== Summary ===")
    print(f"Models fitted (status=ok): {n_fitted}")
    print(f"Time taken: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Output CSV: {csv_path.resolve()}")
    _print_top_models(results_df)

    issues, verify_warnings = _verify_results(results_df, csv_path, expected_models)
    if verify_warnings:
        print("\nVerification warnings:")
        for item in verify_warnings:
            print(f"  - {item}")
    if issues:
        print("\nVerification issues:")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("\nVerification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
