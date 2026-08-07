#!/usr/bin/env python3
"""
Verify REMShapeShiftCalculator and end-to-end changepoint integration.

fill_first=False (default): output length is window_days - 1 (7 for an 8-day window).
Use n_chunks=window_days-1 when running the changepoint model with shape_shift.

Usage:
  python scripts/test_shape_shift_feature.py
  python scripts/test_shape_shift_feature.py --smoke-mcmc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seismic_pipeline.seismo.rem_shape_shift import (
    REMShapeShiftCalculator,
    _split_row_into_daily_profiles,
    compute_shape_shift_map,
)
from seismic_pipeline.features.rem_chunk_features import build_group_data


def test_calculator_unit() -> None:
    """Synthetic daily profiles: identical days -> zero shift; perturbed -> positive."""
    n_days, n_points = 8, 19
    rng = np.random.default_rng(0)
    day_template = rng.uniform(5.0, 40.0, size=n_points)
    daily = np.tile(day_template, (n_days, 1))
    daily[3] += rng.uniform(1.0, 5.0, size=n_points)

    calc = REMShapeShiftCalculator(fill_first=False)
    shifts = calc.compute(daily)

    assert shifts.shape == (n_days - 1,), shifts.shape
    assert np.all(shifts >= 0), "shape_shift must be non-negative"
    assert shifts[1] < 1e-9, "identical consecutive days should yield ~0 shift"
    assert shifts[2] > 0, "perturbed day should yield positive shift"

    calc_pad = REMShapeShiftCalculator(fill_first=True)
    shifts_pad = calc_pad.compute(daily)
    assert shifts_pad.shape == (n_days,)
    assert shifts_pad[0] == 0.0
    np.testing.assert_allclose(shifts_pad[1:], shifts)

    single = REMShapeShiftCalculator(fill_first=False).compute(daily[:1])
    assert single.shape == (0,)

    print("PASS: REMShapeShiftCalculator unit tests")


def test_split_and_batch() -> None:
    n_days, n_points = 8, 19
    day_vec = np.linspace(10.0, 30.0, n_points)
    profiles = np.tile(day_vec, n_days).reshape(1, -1)
    padded = np.pad(profiles, ((0, 0), (0, 5)), constant_values=np.nan)
    daily = _split_row_into_daily_profiles(padded[0], n_days)
    assert daily.shape == (n_days, n_points)

    batch = compute_shape_shift_map(padded, n_days=n_days, fill_first=False)
    assert batch.shape == (1, n_days - 1)
    assert np.all(np.isfinite(batch))
    assert batch[0, 0] < 1e-9, "identical daily profiles -> zero shift"

    print("PASS: split + batch map tests")


def test_build_group_data_synthetic() -> None:
    n_events, n_days, n_points = 4, 8, 19
    width = n_days * n_points
    data_raw = np.tile(np.arange(width, dtype=float), (n_events, 1))
    data_norm = data_raw / np.max(data_raw)

    group_data = build_group_data(
        data_norm,
        n_chunks=n_days - 1,
        feature_selection={"concat": ["shape_shift"]},
        data_raw=data_raw,
        window_days=n_days,
        n_points_per_day=n_points,
    )
    df = group_data["concat"]["shape_shift"]
    assert df.shape == (n_events, n_days - 1)
    assert list(df.columns)[0].startswith("shape_shift_concat_chunk_")
    col_std = df.std(axis=0)
    print(f"  per-column std across events: {col_std.values}")
    print("PASS: build_group_data synthetic integration")


def run_smoke_mcmc(csv_path: Path | None, window_days: int) -> None:
    from seismic_pipeline.features.rem_chunk_features import prepare_model_data, set_runtime_data_norm
    from seismic_pipeline.features.runtime import set_runtime_export_cfg
    from seismic_pipeline.bayesian import runner as runner_mod
    from seismic_pipeline.bayesian.runner import run_variant

    if csv_path is None or not csv_path.is_file():
        print("SKIP: smoke MCMC (no CSV path; run export first)")
        return

    runner_mod.plot_trace_and_tau = lambda *args, **kwargs: None
    runner_mod.plot_posteriors_like_script = lambda *args, **kwargs: None
    runner_mod.feature_likelihood_profiles = lambda *args, **kwargs: {}

    set_runtime_export_cfg({"window_days": window_days})
    prep = prepare_model_data(str(csv_path))
    set_runtime_data_norm(prep["data_norm"])
    n_chunks = window_days - 1

    trace, summary = run_variant(
        "shape_shift smoke",
        n_chunks=n_chunks,
        feature_selection={"concat": ["shape_shift"]},
        parameter_selection={"shape_shift": {"likelihood": "lognormal"}},
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        plot_likelihood_profiles=False,
        data_norm=prep["data_norm"],
    )
    if summary is not None and not summary.empty:
        print(summary[["mean", "sd", "r_hat"]].to_string())
    try:
        import arviz as az

        idata = az.from_pymc(trace)
        diverging = int(idata.sample_stats.diverging.sum().values)
    except Exception:
        diverging = -1
    print(f"  divergences: {diverging}")
    print("PASS: smoke MCMC completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-mcmc",
        action="store_true",
        help="Run a short run_variant MCMC after unit tests",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="NaN-padded REM profile CSV for smoke MCMC",
    )
    parser.add_argument("--window-days", type=int, default=8)
    args = parser.parse_args()

    print("=== shape_shift feature tests ===\n")
    test_calculator_unit()
    test_split_and_batch()
    test_build_group_data_synthetic()

    if args.smoke_mcmc:
        print()
        run_smoke_mcmc(args.csv, args.window_days)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
