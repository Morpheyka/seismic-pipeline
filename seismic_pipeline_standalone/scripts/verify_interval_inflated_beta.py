#!/usr/bin/env python3
"""Smoke-test Interval-Inflated Beta likelihood with synthetic range data."""
from __future__ import annotations

import os
import sys

# Package lives in seismic_pipeline_standalone/ (parent of scripts/).
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np
import pandas as pd

from seismic_pipeline.bayesian.changepoint_model import (
    build_changepoint_model,
    interval_inflated_beta_logp,
    sample_interval_inflated_beta,
    sample_model,
)
from seismic_pipeline.bayesian.diagnostics import _values_flat, score_changepoint_trace
from seismic_pipeline.bayesian.priors import _parse_parameter_selection


def _group_data(values: np.ndarray) -> dict:
    return {"concat": {"range": pd.DataFrame(values)}}


def _run_case(name: str, values: np.ndarray, *, expect_high_pi: bool) -> None:
    group_data = _group_data(values)
    ps = {"range": {"likelihood": "interval_inflated_beta", "threshold": 0.9}}
    model = build_changepoint_model(
        group_data,
        tau_lower=2,
        tau_upper=6,
        parameter_selection=ps,
        tau_mode="marginalized",
    )
    trace = sample_model(model, draws=400, tune=400, chains=2, cores=1, progressbar=False)
    pi_mean = float(
        np.mean(_values_flat(trace, "pi_concat_range_1"))
        + np.mean(_values_flat(trace, "pi_concat_range_2"))
    ) / 2.0
    score = score_changepoint_trace(trace, group_data=group_data, parameter_selection=ps, model=model)
    ok_pi = pi_mean > 0.3 if expect_high_pi else pi_mean < 0.2
    ok_div = score["n_divergences"] == 0
    print(
        f"{name}: pi_mean={pi_mean:.3f} divergences={score['n_divergences']} "
        f"r_hat_max={score['r_hat_max']:.3f} pi_ok={ok_pi} div_ok={ok_div}"
    )
    if not ok_pi or not ok_div:
        sys.exit(1)


def main() -> None:
    _ = interval_inflated_beta_logp
    cfg = _parse_parameter_selection({"range": {"likelihood": "interval_inflated_beta"}}, {"range"})
    assert cfg["range"]["likelihood"] == "interval_inflated_beta"

    rng = np.random.default_rng(42)
    _run_case("high_pi", rng.uniform(0.92, 0.99, size=(5, 8)), expect_high_pi=True)
    _run_case("low_pi", rng.uniform(0.3, 0.7, size=(5, 8)), expect_high_pi=False)

    beta_data = rng.beta(2, 5, size=(5, 8))
    ps_beta = {"range": {"likelihood": "beta"}}
    model_beta = build_changepoint_model(
        _group_data(beta_data),
        tau_lower=2,
        tau_upper=6,
        parameter_selection=ps_beta,
        tau_mode="marginalized",
    )
    trace_beta = sample_model(model_beta, draws=200, tune=200, chains=2, cores=1, progressbar=False)
    score_beta = score_changepoint_trace(
        trace_beta,
        group_data=_group_data(beta_data),
        parameter_selection=ps_beta,
        model=model_beta,
    )
    print(f"beta_compat: divergences={score_beta['n_divergences']}")
    if score_beta["n_divergences"] > 0:
        sys.exit(1)

    s = sample_interval_inflated_beta(rng, 100, 0.5, 2, 2, 0.9)
    assert s.shape == (100,)
    print("All checks passed.")


if __name__ == "__main__":
    main()
