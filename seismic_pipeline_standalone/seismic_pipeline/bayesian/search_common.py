"""Shared utilities for changepoint model search (exhaustive and MH)."""
from __future__ import annotations

import copy
from typing import Any, List, Tuple

import numpy as np

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, resolve_n_group_chunks, sample_model
from seismic_pipeline.bayesian.diagnostics import _available_varnames, score_changepoint_trace
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    group_data_from_precomputed,
)


from seismic_pipeline.features.runtime import get_runtime_data_raw, get_runtime_export_cfg


def clone_config(config: dict) -> dict:
    return copy.deepcopy(config)


def validate_feature_selection_for_n_chunks(
    feature_selection: dict,
    n_chunks: int,
    *,
    window_days: int | None = None,
    data_raw: np.ndarray | None = None,
) -> bool:
    try:
        wide = max(512, int(n_chunks) * 32)
        raw = data_raw if data_raw is not None else np.zeros((2, wide), dtype=float)
        wd = window_days
        export_cfg = get_runtime_export_cfg()
        if wd is None and export_cfg is not None:
            wd = int(export_cfg.get("window_days", 10))
        build_group_data(
            np.zeros((2, wide), dtype=float),
            data_raw=raw,
            n_chunks=n_chunks,
            feature_selection=feature_selection,
            window_days=wd,
        )
        return True
    except Exception:
        return False


def build_summary_var_names(group_data: dict, trace) -> List[str]:
    available = _available_varnames(trace)
    out: List[str] = []
    if "tau" in available:
        out.append("tau")
    if "tau_mean" in available:
        out.append("tau_mean")
    for group_name, features in group_data.items():
        for feat_name in features:
            for param_name in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in available and p2 in available:
                    out.extend([p1, p2])
            nu_name = f"nu_{group_name}_{feat_name}"
            if nu_name in available:
                out.append(nu_name)
    return sorted(set(out))


def fit_config_once(
    config: dict,
    *,
    data_norm: np.ndarray,
    draws: int,
    tune: int,
    nuts_backend: str,
    chains: int,
    cores: int | None,
    tau_mode: str,
    tau_lower: int,
    tau_upper: int | None,
    ic_criterion: str = "waic",
    sampler_progressbar: bool = True,
    precomputed_features: dict[tuple, np.ndarray] | None = None,
    data_raw: np.ndarray | None = None,
    window_days: int | None = None,
) -> Tuple[dict, Any, dict, Any]:
    """Build group_data, sample, return (group_data, trace, score_parts, model)."""
    if precomputed_features is not None:
        group_data = group_data_from_precomputed(precomputed_features, config)
    else:
        raw = data_raw if data_raw is not None else get_runtime_data_raw()
        wd = window_days
        export_cfg = get_runtime_export_cfg()
        if wd is None and export_cfg is not None:
            wd = int(export_cfg.get("window_days", 10))
        group_data = build_group_data(
            data_norm,
            n_chunks=int(config["n_chunks"]),
            feature_selection=config["feature_selection"],
            data_raw=raw,
            window_days=wd,
        )
    n_group_chunks = resolve_n_group_chunks(group_data)
    tu = tau_upper if tau_upper is not None else n_group_chunks
    model = build_changepoint_model(
        group_data,
        tau_lower=tau_lower,
        tau_upper=tu,
        parameter_selection=config.get("parameter_selection"),
        tau_mode=tau_mode,
    )
    trace = sample_model(
        model,
        draws=draws,
        tune=tune,
        nuts_backend=nuts_backend,
        chains=chains,
        cores=cores,
        progressbar=sampler_progressbar,
    )
    summ_vars = build_summary_var_names(group_data, trace)
    score_parts = score_changepoint_trace(
        trace,
        group_data=group_data,
        parameter_selection=config.get("parameter_selection"),
        tau_threshold=float(config.get("tau_threshold", 7.0)),
        summary_var_names=summ_vars if summ_vars else None,
        model=model,
        criterion=ic_criterion,
        warn_on_fallback=False,
    )
    return group_data, trace, score_parts, model


# Backward-compatible private aliases used by search.py / mh_search.py
_clone_config = clone_config
_validate_feature_selection_for_n_chunks = validate_feature_selection_for_n_chunks
_build_summary_var_names = build_summary_var_names
_fit_config_once = fit_config_once
