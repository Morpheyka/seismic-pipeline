"""
Backward-compatible facade for the split ``seismic_pipeline`` changepoint modules.

Legacy notebooks may continue ``import rem_profiles_export_10days_lib as remlib``.
"""

from __future__ import annotations

from seismic_pipeline.config.changepoint_defaults import (
    DEFAULT_EVENTS_10D,
    DEFAULT_S3_CONFIG,
    FEATURE_SELECTION_PRESETS,
)
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    compute_chunk_feature_map,
    compute_chunk_features,
    compute_concat_chunk_feature_map,
    compute_chunk_feature_map_fixed_n,
    compute_shape_shift,
    expected_fixed_n_chunk_count,
    expected_fixed_n_chunks,
    group_data_from_precomputed,
    load_and_normalize,
    maxmin_scale,
    precompute_all_features,
    prepare_model_data,
    prepare_variant_data,
    set_runtime_data_norm,
)
from seismic_pipeline.features.runtime import (
    ChangepointRunContext,
    get_context,
    get_runtime_export_cfg,
    get_runtime_prepare_cfg,
)
from seismic_pipeline.bayesian.priors import parameter_selection_with_g_prior
from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import (
    _available_varnames,
    _collect_pareto_k_stats,
    _idata_for_waic_from_trace,
    changepoint_log_target,
    changepoint_model_config_fingerprint,
    collect_pareto_k_stats,
    feature_likelihood_profiles,
    idata_for_waic_from_trace,
    p_tau_gt_from_trace,
    score_changepoint_trace,
    summary_from_trace,
    tau_probabilities,
)
from seismic_pipeline.bayesian.search import (
    compute_model_distance_matrix,
    exhaustive_model_search,
    exhaustive_tau_map_table,
    export_exhaustive_search_results_to_csv,
    model_config_hamming_distance,
    print_exhaustive_tau_map_table,
    propose_changepoint_model_config,
    select_exhaustive_top_configs,
    summarize_exhaustive_search,
)
from seismic_pipeline.bayesian.mh_search import (
    check_mh_convergence,
    metropolis_hastings_model_search,
    summarize_model_search,
)
from seismic_pipeline.bayesian.runner import run_variant, plot_posterior_predictive_check
from seismic_pipeline.visualization.changepoint_plots import (
    plot_exhaustive_search_results,
    plot_model_search_results,
    plot_posteriors_like_script,
    plot_trace_and_tau,
)

# Legacy runtime globals (shared via features.runtime)
from seismic_pipeline.features.runtime import (
    _RUNTIME_DATA_NORM,
    _RUNTIME_LAST_EXPORT_CFG,
    _RUNTIME_LAST_PREPARE_CFG,
)

__all__ = [
    "DEFAULT_EVENTS_10D",
    "DEFAULT_S3_CONFIG",
    "export_rem_profiles_10days_cached_only",
    "maxmin_scale",
    "load_and_normalize",
    "compute_chunk_feature_map",
    "compute_concat_chunk_feature_map",
    "compute_chunk_feature_map_fixed_n",
    "compute_shape_shift",
    "expected_fixed_n_chunk_count",
    "expected_fixed_n_chunks",
    "compute_chunk_features",
    "prepare_variant_data",
    "build_group_data",
    "precompute_all_features",
    "group_data_from_precomputed",
    "prepare_model_data",
    "set_runtime_data_norm",
    "FEATURE_SELECTION_PRESETS",
    "parameter_selection_with_g_prior",
    "build_changepoint_model",
    "sample_model",
    "summary_from_trace",
    "tau_probabilities",
    "feature_likelihood_profiles",
    "plot_trace_and_tau",
    "plot_posteriors_like_script",
    "changepoint_model_config_fingerprint",
    "score_changepoint_trace",
    "changepoint_log_target",
    "propose_changepoint_model_config",
    "check_mh_convergence",
    "metropolis_hastings_model_search",
    "exhaustive_model_search",
    "model_config_hamming_distance",
    "compute_model_distance_matrix",
    "summarize_exhaustive_search",
    "export_exhaustive_search_results_to_csv",
    "exhaustive_tau_map_table",
    "print_exhaustive_tau_map_table",
    "select_exhaustive_top_configs",
    "p_tau_gt_from_trace",
    "plot_exhaustive_search_results",
    "summarize_model_search",
    "plot_model_search_results",
    "plot_posterior_predictive_check",
    "run_variant",
    "collect_pareto_k_stats",
    "idata_for_waic_from_trace",
    "get_runtime_export_cfg",
    "get_runtime_prepare_cfg",
    "ChangepointRunContext",
    "get_context",
]
