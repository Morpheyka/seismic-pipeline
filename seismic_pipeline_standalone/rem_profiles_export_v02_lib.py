"""
Backward-compatible shim for ``rem_profiles_export_v02_lib``.

Re-exports the core export → prepare → fit API from the split package.
"""

from rem_profiles_export_10days_lib import (
    build_changepoint_model,
    build_group_data,
    compute_chunk_features,
    export_rem_profiles_10days_cached_only,
    plot_posteriors_like_script,
    plot_trace_and_tau,
    prepare_model_data,
    run_variant,
    sample_model,
    set_runtime_data_norm,
    summary_from_trace,
    tau_probabilities,
)

__all__ = [
    "export_rem_profiles_10days_cached_only",
    "compute_chunk_features",
    "prepare_model_data",
    "build_group_data",
    "set_runtime_data_norm",
    "build_changepoint_model",
    "sample_model",
    "summary_from_trace",
    "tau_probabilities",
    "plot_trace_and_tau",
    "plot_posteriors_like_script",
    "run_variant",
]
