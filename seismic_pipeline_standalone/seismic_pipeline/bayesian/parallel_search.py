"""Parallel exhaustive search with profile-level caching and memory management."""
from __future__ import annotations

import gc
import json
import math
import multiprocessing
import os
import traceback
import warnings
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Any, Optional

# Configure BLAS/OpenMP before importing numpy/pymc stack.
for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "10")

import arviz as az
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from seismic_pipeline.bayesian.changepoint_model import (
    build_changepoint_model,
    sample_model,
)
from seismic_pipeline.bayesian.diagnostics import (
    changepoint_model_config_fingerprint,
    collect_pareto_k_stats,
    idata_for_waic_from_trace,
    p_tau_gt_from_trace,
    score_changepoint_trace,
)
from seismic_pipeline.bayesian.search_common import build_scoring_summary_var_names
from seismic_pipeline.config.changepoint_defaults import (
    LIKELIHOOD_CHOICES_BY_METRIC,
    REM_PROFILE_CHOICES,
    validate_rem_profile_params,
)
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    expected_fixed_n_chunk_count,
    prepare_model_data,
)
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only

_SCORING_POSTERIOR_VARS: tuple[str, ...] = (
    "changepoint_pointwise_log_lik",
    "tau_probs",
    "tau_support",
)
_AFFINITY_LOGGED: set[int] = set()


@dataclass
class ParallelSearchConfig:
    """Configuration for parallel exhaustive search."""

    # REM profile grid.
    n_points_choices: list[int] = field(default_factory=lambda: [12, 24, 48])
    overlap_choices: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5])

    # Feature grid (max K active group/metric blocks per config).
    max_features: int = 3
    feature_groups: list[str] = field(default_factory=lambda: ["concat", "even", "odd"])
    # n_chunks mode for candidate configs:
    # - "fixed_halfday_chunks": n_days * (n_points_per_day // 2) [legacy concat/even/odd]
    # - "window_days": n_days [daily/day/night aggregations]
    # - "shape_shift": n_days - 1 [day-to-day L1 distance feature]
    n_chunks_mode: str = "fixed_halfday_chunks"
    feature_metrics: list[str] = field(default_factory=lambda: ["mean", "range"])

    # Likelihood families.
    mean_likelihoods: list[str] = field(
        default_factory=lambda: list(LIKELIHOOD_CHOICES_BY_METRIC["mean"])
    )
    range_likelihoods: list[str] = field(
        default_factory=lambda: list(LIKELIHOOD_CHOICES_BY_METRIC["range"])
    )
    shape_shift_likelihoods: list[str] = field(
        default_factory=lambda: list(LIKELIHOOD_CHOICES_BY_METRIC["shape_shift"])
    )

    # MCMC settings.
    draws: int = 500
    tune: int = 1000
    chains: int = 4
    # BLAS/OpenMP threads per MCMC chain process (passed as blas_cores=chains * this).
    cores_per_chain: int = 1
    # "pymc" (default), "numpyro", or "blackjax". JAX backends need tau_mode="marginalized".
    nuts_backend: str = "blackjax"
    # NumPyro chain layout: "auto", "parallel", or "vectorized".
    # "auto" picks "parallel" (fastest on CPU in our benchmarks).
    jax_chain_method: str = "auto"
    tau_mode: str = "marginalized"
    tau_lower: int = 3
    tau_upper: int = 8
    tau_threshold: float = 5.0

    # Parallelism.
    n_jobs: int = 4

    # Memory.
    gc_frequency: int = 10
    blas_total_cores: int = 10
    # Explicit BLAS/OpenMP threads per worker process; if None, derive from blas_total_cores.
    blas_threads_per_worker: int | None = None
    # Worker CPU affinity mode: "none", "single_core_smt", "numa_spread", "numa_partition".
    worker_affinity_mode: str = "none"
    # Keep this many physical cores free on each NUMA node in "numa_partition" mode.
    reserve_physical_cores_per_numa_node: int = 0
    # If True and cores_per_chain>=2, pin each worker to SMT siblings of one core.
    pin_blas_to_single_core: bool = False
    # If True, spread workers across NUMA nodes (sockets) by CPU affinity.
    numa_spread_workers: bool = False

    # Pareto-k.
    pareto_threshold: float = 0.7
    record_pareto_events: bool = True
    max_pareto_retries: int = 2

    # Paths.
    out_dir: Path = Path("./run_output_8day_parallel")

    # Data export.
    window_days: int = 8
    rem_stage: int = 2

    # Day-mask (artifacts ∪ missing); K = min_valid_days.
    day_mask: bool = False
    day_mask_apply_artifacts: bool = True
    min_valid_days: int = 6
    # Plain beta is diagnostic-only when True (excluded from rank_eligible).
    plain_beta_diagnostic_only: bool = False

    # Convergence filters.
    rhat_threshold: float = 1.05
    ess_threshold: float = 100.0
    # Optional fail-fast gate: skip LOO/Pareto for clearly bad chains.
    fail_fast_on_diagnostics: bool = False
    fail_fast_max_divergences: int = 0
    fail_fast_rhat_threshold: float | None = 1.05
    fail_fast_ess_threshold: float | None = 100.0

    @property
    def rem_profile_grid(self) -> list[dict[str, int | float]]:
        """Validated REM profile grid from n_points_choices/overlap_choices."""
        out: list[dict[str, int | float]] = []
        for n_points in self.n_points_choices:
            for overlap in self.overlap_choices:
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


def run_parallel_search(
    config: ParallelSearchConfig,
    export_base_cfg: dict,
    resume_from_csv: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run profile-cached exhaustive search with joblib parallel workers."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "profile_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = out_dir / "exhaustive_search_parallel.checkpoint.csv"

    if "events" not in export_base_cfg:
        raise ValueError("export_base_cfg must include 'events' for REM export.")

    nuts_backend = str(config.nuts_backend).strip().lower()
    if nuts_backend in {"numpyro", "blackjax"} and str(config.tau_mode).strip().lower() != "marginalized":
        raise ValueError(
            f"nuts_backend={nuts_backend!r} requires tau_mode='marginalized' "
            f"(got {config.tau_mode!r})."
        )

    _configure_blas_threads(config)

    completed_fingerprints: set[str] = set()
    all_results: list[dict[str, Any]] = []
    if resume_from_csv is not None and Path(resume_from_csv).exists():
        existing = pd.read_csv(Path(resume_from_csv))
        completed_fingerprints = {
            str(fp) for fp in existing.get("fingerprint", pd.Series(dtype=str)).dropna().tolist()
        }
        all_results.extend(existing.to_dict(orient="records"))
        if verbose:
            print(f"[parallel-search] resume: {len(completed_fingerprints)} fingerprints preloaded")

    profile_grid = config.rem_profile_grid
    profile_iter: Any = profile_grid
    if verbose and tqdm is not None:
        profile_iter = tqdm(profile_grid, desc="Profiles", unit="profile")
    for profile_idx, rem_params in enumerate(profile_iter, start=1):
        n_points = int(rem_params["n_points_per_day"])
        overlap = float(rem_params["overlap"])
        rem_stage = int(rem_params["rem_stage"])
        if verbose:
            print(
                f"\n[parallel-search] profile {profile_idx}/{len(profile_grid)} "
                f"n_points={n_points} overlap={overlap:.2f} rem_stage={rem_stage}",
                flush=True,
            )

        profile_out_dir = cache_root / (
            f"rem_n{n_points}_ov{overlap:.2f}_stage{rem_stage}"
        )
        profile_out_dir.mkdir(parents=True, exist_ok=True)

        export_cfg = dict(export_base_cfg)
        export_cfg["output_dir"] = str(profile_out_dir)
        export_cfg["n_points_per_day"] = n_points
        export_cfg["overlap"] = overlap
        export_cfg["rem_stage"] = rem_stage
        export_cfg.setdefault("window_days", int(config.window_days))

        export_result = export_rem_profiles_10days_cached_only(**export_cfg)
        prep = prepare_model_data(
            csv_path=export_result["paths"]["nanpad_output_csv"],
            day_mask=bool(config.day_mask),
            apply_artifacts=bool(config.day_mask_apply_artifacts),
            min_valid_days=int(config.min_valid_days),
            n_points_per_day=int(n_points),
            window_days=int(config.window_days),
        )
        data_norm = np.asarray(prep["data_norm"], dtype=float)
        data_raw = np.asarray(prep["data_raw"], dtype=float)
        good_indices = np.asarray(prep["good_indices"], dtype=int)
        day_valid = prep.get("day_valid")
        if day_valid is not None:
            day_valid = np.asarray(day_valid, dtype=bool)
        if verbose and bool(config.day_mask):
            print(
                f"[parallel-search] day_mask ON: n_events={data_norm.shape[0]} "
                f"n_masked_days={prep.get('n_masked_days', 0)} "
                f"apply_artifacts={config.day_mask_apply_artifacts}",
                flush=True,
            )
        n_chunks_mode = str(config.n_chunks_mode).strip().lower()
        if n_chunks_mode == "window_days":
            n_chunks = int(config.window_days)
        elif n_chunks_mode == "shape_shift":
            n_chunks = int(config.window_days) - 1
        else:
            n_chunks = expected_fixed_n_chunk_count(
                n_points_per_day=n_points,
                n_days=int(config.window_days),
            )

        feature_configs = _generate_feature_configs(
            max_features=config.max_features,
            feature_groups=config.feature_groups,
            feature_metrics=config.feature_metrics,
        )
        model_configs: list[dict[str, Any]] = []
        for feat_cfg in feature_configs:
            for param_sel in _generate_likelihood_combos(
                feature_selection=feat_cfg,
                mean_likelihoods=config.mean_likelihoods,
                range_likelihoods=config.range_likelihoods,
                shape_shift_likelihoods=config.shape_shift_likelihoods,
            ):
                cfg = {
                    "rem_profile_params": dict(rem_params),
                    "n_chunks": int(n_chunks),
                    "feature_selection": feat_cfg,
                    "parameter_selection": param_sel,
                    "tau_threshold": float(config.tau_threshold),
                }
                fp = changepoint_model_config_fingerprint(cfg)
                if fp not in completed_fingerprints:
                    model_configs.append(cfg)

        if verbose:
            print(
                f"[parallel-search] profile models to fit: {len(model_configs)} "
                f"(already done: {len(completed_fingerprints)})",
                flush=True,
            )

        profile_results = _fit_models_parallel(
            configs=model_configs,
            data_norm=data_norm,
            data_raw=data_raw,
            good_indices=good_indices,
            n_points=n_points,
            search_config=config,
            day_valid=day_valid,
            verbose=verbose,
            progress_desc=(
                f"Models n={n_points} ov={overlap:.2f}"
                if verbose
                else None
            ),
        )
        all_results.extend(profile_results)
        completed_fingerprints.update(
            str(rec.get("fingerprint")) for rec in profile_results if rec.get("fingerprint")
        )
        if all_results:
            pd.DataFrame(all_results).to_csv(checkpoint_csv, index=False)
            if verbose:
                print(f"[parallel-search] checkpoint saved: {checkpoint_csv}", flush=True)

        del data_norm
        del data_raw
        del prep
        gc.collect()
        _clear_pytensor_cache()

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        sort_key = pd.to_numeric(
            results_df.get("elpd_loo_per_feature_event", pd.Series(dtype=float)),
            errors="coerce",
        )
        eligible_key = results_df.get("rank_eligible", pd.Series(False, index=results_df.index))
        eligible_key = eligible_key.fillna(False).astype(bool).astype(int)
        results_df = results_df.assign(_sort_eligible=eligible_key, _sort_loo=sort_key)
        results_df = results_df.sort_values(
            ["_sort_eligible", "_sort_loo"],
            ascending=[False, False],
        ).drop(columns=["_sort_eligible", "_sort_loo"])
        results_df = results_df.reset_index(drop=True)
        results_df["rank_by_loo"] = np.arange(1, len(results_df) + 1, dtype=int)

    output_csv = out_dir / "exhaustive_search_parallel.csv"
    results_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"[parallel-search] saved: {output_csv}", flush=True)
    return results_df


def _generate_feature_configs(
    *,
    max_features: int = 3,
    feature_groups: list[str] | tuple[str, ...] = ("concat", "even", "odd"),
    feature_metrics: list[str] | tuple[str, ...] = ("mean", "range"),
) -> list[dict[str, list[str]]]:
    """Generate all non-empty feature combinations up to max_features blocks."""
    if max_features < 1:
        raise ValueError("max_features must be >= 1")
    groups = [str(g).strip().lower() for g in feature_groups if str(g).strip()]
    if not groups:
        raise ValueError("feature_groups must contain at least one group name.")
    metrics = [str(m).strip().lower() for m in feature_metrics if str(m).strip()]
    if not metrics:
        raise ValueError("feature_metrics must contain at least one metric name.")
    all_features = [(group_name, metric_name) for group_name in groups for metric_name in metrics]
    configs: list[dict[str, list[str]]] = []
    for k in range(1, max_features + 1):
        for combo in combinations(all_features, k):
            feat_dict: dict[str, list[str]] = {}
            for group, metric in combo:
                feat_dict.setdefault(group, [])
                if metric not in feat_dict[group]:
                    feat_dict[group].append(metric)
            normalized = {group: sorted(metrics) for group, metrics in sorted(feat_dict.items())}
            configs.append(normalized)
    return configs


def _generate_likelihood_combos(
    feature_selection: dict[str, list[str]],
    mean_likelihoods: list[str],
    range_likelihoods: list[str],
    shape_shift_likelihoods: list[str],
) -> list[dict[str, dict[str, str]]]:
    """Generate metric-level likelihood combinations for a feature selection."""
    active_metrics = sorted({metric for metrics in feature_selection.values() for metric in metrics})
    if not active_metrics:
        return [{}]

    lik_by_metric: dict[str, list[str]] = {}
    for metric in active_metrics:
        if metric == "mean":
            opts = [str(x).strip().lower() for x in mean_likelihoods]
        elif metric == "range":
            opts = [str(x).strip().lower() for x in range_likelihoods]
        elif metric == "shape_shift":
            opts = [str(x).strip().lower() for x in shape_shift_likelihoods]
        else:
            raise ValueError(f"Unsupported metric '{metric}' in feature_selection.")
        opts = sorted({x for x in opts if x})
        if not opts:
            raise ValueError(f"No likelihood options provided for metric '{metric}'.")
        lik_by_metric[metric] = opts

    combos: list[dict[str, dict[str, str]]] = []
    combo_values = [lik_by_metric[m] for m in active_metrics]
    for values in product(*combo_values):
        spec: dict[str, dict[str, Any]] = {}
        for metric, likelihood in zip(active_metrics, values):
            entry: dict[str, Any] = {"likelihood": likelihood}
            if metric == "range" and likelihood in {
                "beta",
                "beta_constrained",
                "interval_inflated_beta",
                "zero_inflated_beta",
            }:
                # Range on globally normalized [-1, 1] profiles is bounded by [0, 2].
                entry["support_upper"] = 2.0
            if metric == "range" and likelihood == "interval_inflated_beta":
                entry.setdefault("threshold", 0.9)
            spec[metric] = entry
        combos.append(spec)
    return combos


def _fit_models_parallel(
    configs: list[dict[str, Any]],
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    good_indices: np.ndarray,
    n_points: int,
    search_config: ParallelSearchConfig,
    verbose: bool = True,
    progress_desc: str | None = None,
    day_valid: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Fit models in parallel by batches of search_config.n_jobs."""
    if not configs:
        return []
    n_total = len(configs)
    results: list[dict[str, Any]] = []
    completed = 0
    next_gc_mark = int(max(1, search_config.gc_frequency))
    batch_size = max(1, int(search_config.n_jobs))
    pbar = None
    if verbose and tqdm is not None:
        pbar = tqdm(
            total=n_total,
            desc=progress_desc or "Models",
            unit="model",
            leave=False,
        )

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_configs = configs[batch_start:batch_end]
        batch_results = Parallel(
            n_jobs=len(batch_configs),
            backend="loky",
            verbose=0,
        )(
            delayed(_fit_one_model)(
                config=cfg,
                data_norm=data_norm,
                data_raw=data_raw,
                good_indices=good_indices,
                n_points=n_points,
                search_config=search_config,
                day_valid=day_valid,
            )
            for cfg in batch_configs
        )
        results.extend(batch_results)
        completed += len(batch_configs)
        if pbar is not None:
            pbar.update(len(batch_configs))
        while completed >= next_gc_mark:
            gc.collect()
            _clear_pytensor_cache()
            if verbose:
                print(
                    f"[parallel-search] GC cleanup at {completed}/{n_total} models",
                    flush=True,
                )
            next_gc_mark += int(max(1, search_config.gc_frequency))
    if pbar is not None:
        pbar.close()
    return results


def _fit_one_model(
    *,
    config: dict[str, Any],
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    good_indices: np.ndarray,
    n_points: int,
    search_config: ParallelSearchConfig,
    day_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one config with optional iterative Pareto-k event removal."""
    _pin_worker_cpu_affinity_if_needed(search_config)
    fingerprint = changepoint_model_config_fingerprint(config)
    n_active_features = int(sum(len(vals or []) for vals in (config.get("feature_selection") or {}).values()))
    n_days_window = int(search_config.window_days)
    max_retries = max(0, int(search_config.max_pareto_retries))
    active_idx = np.arange(int(data_norm.shape[0]), dtype=int)
    removed_event_indices: list[int] = []
    pareto_history: list[dict[str, Any]] = []
    chains = max(1, int(search_config.chains))
    nuts_backend = str(search_config.nuts_backend).strip().lower()
    try:
        n_chunks = int(config["n_chunks"])
        feature_selection = config["feature_selection"]
        parameter_selection = config["parameter_selection"]
        for attempt_idx in range(max_retries + 1):
            if int(active_idx.size) <= 0:
                raise RuntimeError("No events left after Pareto filtering.")

            data_norm_work = data_norm[active_idx]
            data_raw_work = data_raw[active_idx]
            good_idx_work = np.asarray(good_indices, dtype=int)[active_idx]
            day_valid_work = (
                np.asarray(day_valid, dtype=bool)[active_idx]
                if day_valid is not None
                else None
            )
            n_model_events = int(data_norm_work.shape[0])
            n_model_days = int(n_model_events * n_days_window)
            loo_norm_den = int(max(1, n_active_features) * max(1, n_model_days))

            group_data = build_group_data(
                data_norm_work,
                n_chunks=n_chunks,
                feature_selection=feature_selection,
                data_raw=data_raw_work,
                window_days=n_days_window,
                n_points_per_day=int(n_points),
                day_valid=day_valid_work,
            )
            tau_upper = (
                int(search_config.tau_upper)
                if search_config.tau_upper is not None
                else int(n_chunks)
            )
            model = build_changepoint_model(
                group_data,
                tau_lower=int(search_config.tau_lower),
                tau_upper=tau_upper,
                parameter_selection=parameter_selection,
                tau_mode=str(search_config.tau_mode),
            )
            sample_kwargs: dict[str, object] = {
                "draws": int(search_config.draws),
                "tune": int(search_config.tune),
                "nuts_backend": nuts_backend,
                "chains": chains,
                "progressbar": False,
            }
            if nuts_backend == "pymc":
                sample_kwargs["cores"] = chains
                blas_per_chain = max(1, int(search_config.cores_per_chain))
                sample_kwargs["blas_cores"] = chains * blas_per_chain
            else:
                sample_kwargs["jax_chain_method"] = _resolve_jax_chain_method(search_config)
                sample_kwargs["jax_var_names"] = _SCORING_POSTERIOR_VARS
                sample_kwargs["materialize_posterior_vars"] = _SCORING_POSTERIOR_VARS
            trace = sample_model(model, **sample_kwargs)

            summary_vars = build_scoring_summary_var_names(trace)
            if bool(search_config.fail_fast_on_diagnostics):
                precheck = score_changepoint_trace(
                    trace,
                    group_data=group_data,
                    parameter_selection=parameter_selection,
                    tau_threshold=float(config.get("tau_threshold", search_config.tau_threshold)),
                    summary_var_names=summary_vars if summary_vars else None,
                    model=None,
                    criterion="loo",
                    warn_on_fallback=False,
                    loo_report="elpd",
                )
                ff_reasons = _fail_fast_reasons(precheck, search_config=search_config)
                if ff_reasons:
                    record = {
                        "fingerprint": fingerprint,
                        "status": "diagnostic_reject",
                        "error": None,
                        "elpd_loo": float("nan"),
                        "loo_ic": float("nan"),
                        "loo_reported": float("nan"),
                        "loo": float("nan"),
                        "elpd_loo_per_event": float("nan"),
                        "elpd_loo_per_feature": float("nan"),
                        "elpd_loo_per_feature_event": float("nan"),
                        "elpd_loo_per_feature_day": float("nan"),
                        "n_model_events_final": n_model_events,
                        "n_model_days_final": n_model_days,
                        "loo_norm_denominator": loo_norm_den,
                        "r_hat_max": float(precheck.get("r_hat_max", float("nan"))),
                        "ess_min_bulk": float(precheck.get("ess_min_bulk", float("nan"))),
                        "ess_min_tail": float(precheck.get("ess_min_tail", float("nan"))),
                        "n_divergences": int(precheck.get("n_divergences", 0)),
                        "bfmi": float(precheck.get("bfmi", precheck.get("bfmi_approx", float("nan")))),
                        "tau_map": int(precheck.get("map_tau", -1)),
                        "e_tau": float(precheck.get("e_tau", float("nan"))),
                        "tau_std": float(precheck.get("tau_std", float("nan"))),
                        "tau_q1": float(precheck.get("tau_q1", float("nan"))),
                        "tau_q2": float(precheck.get("tau_q2", float("nan"))),
                        "tau_q3": float(precheck.get("tau_q3", float("nan"))),
                        "tau_hdi_60_lower": float(precheck.get("tau_hdi_60_lower", float("nan"))),
                        "tau_hdi_60_upper": float(precheck.get("tau_hdi_60_upper", float("nan"))),
                        "tau_hdi_60_width": float(precheck.get("tau_hdi_60_width", float("nan"))),
                        "p_tau_gt_6": float(p_tau_gt_from_trace(trace, 6.0)),
                        "loo_pareto_k_max": float("nan"),
                        "n_over_threshold": 0,
                        "loo_n_over_threshold": 0,
                        "pareto_retry_count": int(len(removed_event_indices)),
                        "pareto_unresolved_after_retries": False,
                        "pareto_refit_history_json": json.dumps(pareto_history),
                        "pareto_k_values": "[]",
                        "influential_event_indices": "[]",
                        "removed_event_indices": json.dumps([int(x) for x in removed_event_indices]),
                        "good_indices": json.dumps(good_idx_work.tolist()),
                        "n_active_features": n_active_features,
                        "n_model_events": n_model_events,
                        "features": _format_features(feature_selection),
                        "likelihoods": _format_likelihoods(parameter_selection),
                        "n_points": int(config["rem_profile_params"]["n_points_per_day"]),
                        "overlap": float(config["rem_profile_params"]["overlap"]),
                        "config_json": json.dumps(config, sort_keys=True, default=str),
                        "metric_validation_passed": True,
                        "metric_validation_issues": "[]",
                        "rank_eligible": False,
                        "diagnostic_status": "ineligible",
                        "diagnostic_issues": json.dumps(ff_reasons),
                    }
                    del trace
                    del model
                    gc.collect()
                    return record

            score_parts = score_changepoint_trace(
                trace,
                group_data=group_data,
                parameter_selection=parameter_selection,
                tau_threshold=float(config.get("tau_threshold", search_config.tau_threshold)),
                summary_var_names=summary_vars if summary_vars else None,
                model=model,
                criterion="loo",
                warn_on_fallback=False,
                loo_report="elpd",
            )
            loo_k_max, loo_n_over, influential_event_indices, worst_local_idx = collect_pareto_k_stats(
                trace,
                model,
                pareto_threshold=float(search_config.pareto_threshold),
                idata_ic=score_parts.get("_idata_ic"),
                loo_obj=score_parts.get("_loo_obj"),
            )
            influential_global = [
                int(good_idx_work[int(i)]) for i in (influential_event_indices or []) if int(i) < int(good_idx_work.size)
            ]
            pareto_k_values = _pareto_k_values(loo_obj=score_parts.get("_loo_obj"))
            elpd_loo = float(score_parts.get("elpd_loo", float("nan")))
            loo_ic = float(score_parts.get("loo_ic", float("nan")))
            loo_reported = float(score_parts.get("loo_reported", float("nan")))
            loo_legacy = float(score_parts.get("loo", float("nan")))
            elpd_loo_per_event = elpd_loo / float(n_model_events) if n_model_events > 0 else float("nan")
            elpd_loo_per_feature_event = (
                elpd_loo / float(n_active_features * n_model_events)
                if n_active_features > 0 and n_model_events > 0
                else float("nan")
            )
            elpd_loo_per_feature_day = (
                elpd_loo / float(loo_norm_den)
                if n_active_features > 0 and n_model_days > 0
                else float("nan")
            )

            history_row: dict[str, Any] = {
                "attempt_idx": int(attempt_idx),
                "n_model_events": int(n_model_events),
                "n_model_days": int(n_model_days),
                "kept_event_indices": [int(x) for x in good_idx_work.tolist()],
                "removed_event_indices": [int(x) for x in removed_event_indices],
                "elpd_loo": elpd_loo,
                "loo_ic": loo_ic,
                "loo_reported": loo_reported,
                "elpd_loo_per_feature_day": elpd_loo_per_feature_day,
                "loo_pareto_k_max": float(loo_k_max),
                "loo_n_over_threshold": int(loo_n_over),
                "influential_event_indices": [int(x) for x in influential_global],
                "removed_for_next_attempt": None,
            }
            should_retry = (
                int(loo_n_over) > 0
                and math.isfinite(float(loo_k_max))
                and float(loo_k_max) > float(search_config.pareto_threshold)
                and int(attempt_idx) < int(max_retries)
                and worst_local_idx is not None
                and int(active_idx.size) > 1
            )

            if should_retry:
                drop_local = int(worst_local_idx)
                drop_global = int(good_idx_work[drop_local])
                history_row["removed_for_next_attempt"] = int(drop_global)
                pareto_history.append(history_row)
                removed_event_indices.append(drop_global)
                active_idx = np.delete(active_idx, drop_local)
                del trace
                del model
                gc.collect()
                continue

            pareto_unresolved = bool(
                int(loo_n_over) > 0
                and math.isfinite(float(loo_k_max))
                and float(loo_k_max) > float(search_config.pareto_threshold)
            )
            pareto_history.append(history_row)

            record = {
                "fingerprint": fingerprint,
                "status": "ok",
                "error": None,
                "elpd_loo": elpd_loo,
                "loo_ic": loo_ic,
                "loo_reported": loo_reported,
                "loo": loo_legacy,
                "elpd_loo_per_event": elpd_loo_per_event,
                "elpd_loo_per_feature": elpd_loo_per_feature_event,
                "elpd_loo_per_feature_event": elpd_loo_per_feature_event,
                "elpd_loo_per_feature_day": elpd_loo_per_feature_day,
                "n_model_events_final": n_model_events,
                "n_model_days_final": n_model_days,
                "loo_norm_denominator": loo_norm_den,
                "r_hat_max": float(score_parts.get("r_hat_max", float("nan"))),
                "ess_min_bulk": float(score_parts.get("ess_min_bulk", float("nan"))),
                "ess_min_tail": float(score_parts.get("ess_min_tail", float("nan"))),
                "n_divergences": int(score_parts.get("n_divergences", 0)),
                "bfmi": float(score_parts.get("bfmi", score_parts.get("bfmi_approx", float("nan")))),
                "tau_map": int(score_parts.get("map_tau", -1)),
                "e_tau": float(score_parts.get("e_tau", float("nan"))),
                "tau_std": float(score_parts.get("tau_std", float("nan"))),
                "tau_q1": float(score_parts.get("tau_q1", float("nan"))),
                "tau_q2": float(score_parts.get("tau_q2", float("nan"))),
                "tau_q3": float(score_parts.get("tau_q3", float("nan"))),
                "tau_hdi_60_lower": float(score_parts.get("tau_hdi_60_lower", float("nan"))),
                "tau_hdi_60_upper": float(score_parts.get("tau_hdi_60_upper", float("nan"))),
                "tau_hdi_60_width": float(score_parts.get("tau_hdi_60_width", float("nan"))),
                "p_tau_gt_6": float(p_tau_gt_from_trace(trace, 6.0)),
                "loo_pareto_k_max": float(loo_k_max),
                "n_over_threshold": int(loo_n_over),
                "loo_n_over_threshold": int(loo_n_over),
                "pareto_retry_count": int(len(removed_event_indices)),
                "pareto_unresolved_after_retries": bool(pareto_unresolved),
                "pareto_refit_history_json": json.dumps(pareto_history),
                "pareto_k_values": (
                    json.dumps(pareto_k_values)
                    if search_config.record_pareto_events
                    else "[]"
                ),
                "influential_event_indices": (
                    json.dumps([int(x) for x in influential_global])
                    if search_config.record_pareto_events
                    else "[]"
                ),
                "removed_event_indices": json.dumps([int(x) for x in removed_event_indices]),
                "good_indices": json.dumps(good_idx_work.tolist()),
                "n_active_features": n_active_features,
                "n_model_events": n_model_events,
                "features": _format_features(feature_selection),
                "likelihoods": _format_likelihoods(parameter_selection),
                "n_points": int(config["rem_profile_params"]["n_points_per_day"]),
                "overlap": float(config["rem_profile_params"]["overlap"]),
                "config_json": json.dumps(config, sort_keys=True, default=str),
            }
            metric_issues = _validate_record_invariants(
                record,
                chains=chains,
                draws=int(search_config.draws),
                expect_pareto_len=int(n_model_events) if search_config.record_pareto_events else None,
            )
            record["metric_validation_passed"] = bool(not metric_issues)
            record["metric_validation_issues"] = json.dumps(metric_issues)
            rank_eligible, diag_issues = _diagnostic_gate_issues(record, search_config=search_config)
            record["rank_eligible"] = bool(rank_eligible)
            record["diagnostic_status"] = "eligible" if rank_eligible else "ineligible"
            record["diagnostic_issues"] = json.dumps(diag_issues)
            del trace
            del model
            gc.collect()
            return record

        raise RuntimeError("Pareto refit loop terminated unexpectedly.")
    except Exception as exc:
        traceback.print_exc()
        warnings.warn(f"Model {fingerprint} failed: {exc}")
        gc.collect()
        return {
            "fingerprint": fingerprint,
            "status": "failed",
            "error": str(exc),
            "config_json": json.dumps(config, sort_keys=True, default=str),
            "pareto_refit_history_json": json.dumps(pareto_history),
            "elpd_loo_per_feature_event": float("nan"),
            "elpd_loo_per_feature_day": float("nan"),
            "rank_eligible": False,
            "diagnostic_status": "failed",
            "diagnostic_issues": json.dumps(["fit_failed"]),
        }


def _pareto_k_values(*, loo_obj=None, trace=None, model=None) -> list[float]:
    """Return pointwise Pareto-k values from LOO (empty on failure)."""
    try:
        if loo_obj is None:
            if trace is None or model is None:
                return []
            idata = idata_for_waic_from_trace(trace, model)
            loo_obj = az.loo(idata, pointwise=True)
        pareto = getattr(loo_obj, "pareto_k", None)
        if pareto is None:
            return []
        arr = np.asarray(pareto, dtype=float).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return []


def _validate_record_invariants(
    record: dict[str, Any],
    *,
    chains: int,
    draws: int,
    expect_pareto_len: int | None,
) -> list[str]:
    """Return a list of invariant violations for one search record."""
    issues: list[str] = []

    def _f(name: str) -> float:
        return float(record.get(name, float("nan")))

    elpd_loo = _f("elpd_loo")
    n_events = int(record.get("n_model_events", 0))
    n_active = int(record.get("n_active_features", 0))
    per_event = _f("elpd_loo_per_event")
    per_feat_event = _f("elpd_loo_per_feature_event")
    p_tau_gt = _f("p_tau_gt_6")
    hdi_width = _f("tau_hdi_60_width")
    n_div = int(record.get("n_divergences", 0))

    if n_events > 0 and math.isfinite(elpd_loo):
        exp_per_event = elpd_loo / float(n_events)
        if math.isfinite(per_event) and not math.isclose(per_event, exp_per_event, rel_tol=1e-6, abs_tol=1e-6):
            issues.append("elpd_loo_per_event mismatch")
    if n_events > 0 and n_active > 0 and math.isfinite(elpd_loo):
        exp_per_feat_event = elpd_loo / float(n_active * n_events)
        if math.isfinite(per_feat_event) and not math.isclose(
            per_feat_event,
            exp_per_feat_event,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            issues.append("elpd_loo_per_feature_event mismatch")

    if math.isfinite(p_tau_gt) and not (0.0 <= p_tau_gt <= 1.0):
        issues.append("p_tau_gt_6 out of [0,1]")
    if math.isfinite(hdi_width) and hdi_width < 0.0:
        issues.append("tau_hdi_60_width is negative")
    if n_div < 0 or n_div > int(chains * draws):
        issues.append("n_divergences out of [0, chains*draws]")

    if expect_pareto_len is not None:
        try:
            pareto_vals = json.loads(str(record.get("pareto_k_values", "[]")))
            if isinstance(pareto_vals, list) and len(pareto_vals) != int(expect_pareto_len):
                issues.append("pareto_k_values length mismatch")
        except Exception:
            issues.append("pareto_k_values is not valid JSON list")

    return issues


def compare_scoring_records(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, float]:
    """Return absolute diffs for key scoring metrics between two records."""
    keys = (
        "elpd_loo",
        "loo_ic",
        "p_loo",
        "tau_map",
        "e_tau",
        "tau_std",
        "tau_q1",
        "tau_q2",
        "tau_q3",
        "tau_hdi_60_lower",
        "tau_hdi_60_upper",
        "tau_hdi_60_width",
        "p_tau_gt_6",
        "loo_pareto_k_max",
        "n_over_threshold",
    )
    diffs: dict[str, float] = {}
    for k in keys:
        a = float(reference.get(k, float("nan")))
        b = float(candidate.get(k, float("nan")))
        if math.isnan(a) and math.isnan(b):
            continue
        d = float(abs(a - b))
        diffs[k] = d
        if math.isfinite(a) and math.isfinite(b) and not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            diffs[f"{k}__outside_tol"] = d
    return diffs


def _diagnostic_gate_issues(
    record: dict[str, Any],
    *,
    search_config: ParallelSearchConfig,
) -> tuple[bool, list[str]]:
    """Evaluate whether a record is eligible for final ranking."""
    issues: list[str] = []
    if str(record.get("status", "")) != "ok":
        issues.append("status_not_ok")
    if int(record.get("n_divergences", 0)) > 0:
        issues.append("has_divergences")

    pareto_k_max = float(record.get("loo_pareto_k_max", float("nan")))
    if math.isfinite(pareto_k_max) and pareto_k_max > float(search_config.pareto_threshold):
        issues.append("pareto_k_over_threshold")

    rhat = float(record.get("r_hat_max", float("nan")))
    if math.isfinite(rhat) and rhat > float(search_config.rhat_threshold):
        issues.append("rhat_over_threshold")

    ess = float(record.get("ess_min_bulk", float("nan")))
    if math.isfinite(ess) and ess < float(search_config.ess_threshold):
        issues.append("ess_below_threshold")

    if not bool(record.get("metric_validation_passed", True)):
        issues.append("metric_invariant_failed")

    if bool(getattr(search_config, "plain_beta_diagnostic_only", False)):
        param_sel = record.get("parameter_selection") or {}
        if isinstance(param_sel, str):
            try:
                import json as _json

                param_sel = _json.loads(param_sel)
            except Exception:
                param_sel = {}
        range_lik = ""
        if isinstance(param_sel, dict):
            range_lik = str((param_sel.get("range") or {}).get("likelihood", "")).strip().lower()
        # Also check flattened columns if present.
        if not range_lik:
            range_lik = str(record.get("range_likelihood", "")).strip().lower()
        if range_lik == "beta":
            issues.append("plain_beta_diagnostic_only")

    return (len(issues) == 0), issues


def _fail_fast_reasons(
    score_parts: dict[str, Any],
    *,
    search_config: ParallelSearchConfig,
) -> list[str]:
    """Reasons to skip expensive LOO/Pareto and move to next model."""
    reasons: list[str] = []
    n_div = int(score_parts.get("n_divergences", 0))
    if n_div > int(search_config.fail_fast_max_divergences):
        reasons.append("fail_fast_divergences")

    rhat_thr = search_config.fail_fast_rhat_threshold
    if rhat_thr is not None:
        rhat = float(score_parts.get("r_hat_max", float("nan")))
        if math.isfinite(rhat) and rhat > float(rhat_thr):
            reasons.append("fail_fast_rhat")

    ess_thr = search_config.fail_fast_ess_threshold
    if ess_thr is not None:
        ess = float(score_parts.get("ess_min_bulk", float("nan")))
        if math.isfinite(ess) and ess < float(ess_thr):
            reasons.append("fail_fast_ess")
    return reasons


def _resolve_jax_chain_method(search_config: ParallelSearchConfig) -> str:
    """Pick NumPyro chain layout for CPU/GPU and joblib nesting."""
    method = str(search_config.jax_chain_method).strip().lower()
    if method == "auto":
        # "parallel" is much faster on CPU for our models (~10x vs vectorized).
        return "parallel"
    if method not in {"parallel", "vectorized"}:
        raise ValueError(
            f"Invalid jax_chain_method={search_config.jax_chain_method!r}. "
            "Use 'auto', 'parallel', or 'vectorized'."
        )
    return method


def _parse_worker_index() -> int:
    """Best-effort 0-based worker index from loky/multiprocessing process name."""
    name = str(multiprocessing.current_process().name)
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        try:
            return max(0, int(digits) - 1)
        except Exception:
            return 0
    return max(0, int(os.getpid()))


def _parse_cpu_list(text: str) -> list[int]:
    """Parse Linux cpu list syntax like '0,2,4-6' into ints."""
    out: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _thread_sibling_pairs() -> list[tuple[int, int]]:
    """Return unique SMT sibling pairs from Linux sysfs topology."""
    pairs: set[tuple[int, int]] = set()
    base = "/sys/devices/system/cpu"
    try:
        for entry in os.listdir(base):
            if not entry.startswith("cpu"):
                continue
            suffix = entry[3:]
            if not suffix.isdigit():
                continue
            topo = os.path.join(base, entry, "topology", "thread_siblings_list")
            if not os.path.isfile(topo):
                continue
            with open(topo, "r", encoding="utf-8") as fh:
                cpus = _parse_cpu_list(fh.read().strip())
            if len(cpus) < 2:
                continue
            pair = tuple(sorted((int(cpus[0]), int(cpus[1]))))
            pairs.add(pair)
    except Exception:
        return []
    return sorted(pairs)


def _numa_node_cpu_sets() -> list[list[int]]:
    """Return per-NUMA-node CPU lists from Linux sysfs."""
    node_sets: list[tuple[int, list[int]]] = []
    base = "/sys/devices/system/node"
    try:
        for entry in os.listdir(base):
            if not entry.startswith("node"):
                continue
            suffix = entry[4:]
            if not suffix.isdigit():
                continue
            cpulist_path = os.path.join(base, entry, "cpulist")
            if not os.path.isfile(cpulist_path):
                continue
            with open(cpulist_path, "r", encoding="utf-8") as fh:
                cpus = _parse_cpu_list(fh.read().strip())
            if cpus:
                node_sets.append((int(suffix), cpus))
    except Exception:
        return []
    node_sets.sort(key=lambda x: x[0])
    return [cpus for _, cpus in node_sets]


def _pin_worker_cpu_affinity_if_needed(search_config: ParallelSearchConfig) -> None:
    """Pin worker CPU affinity (NUMA spread and/or single-core SMT pinning)."""
    if not hasattr(os, "sched_setaffinity"):
        return

    mode = str(search_config.worker_affinity_mode).strip().lower()
    if not mode or mode == "none":
        # Backward-compatible flags.
        use_single_core = bool(search_config.pin_blas_to_single_core) and int(search_config.cores_per_chain) >= 2
        use_numa_spread = bool(search_config.numa_spread_workers)
        if use_single_core and use_numa_spread:
            mode = "single_core_smt"
        elif use_single_core:
            mode = "single_core_smt"
        elif use_numa_spread:
            mode = "numa_spread"
        else:
            mode = "none"

    if mode not in {"none", "single_core_smt", "numa_spread", "numa_partition"}:
        raise ValueError(
            f"Unknown worker_affinity_mode={search_config.worker_affinity_mode!r}. "
            "Use one of: none, single_core_smt, numa_spread, numa_partition."
        )
    if mode == "none":
        return

    idx = _parse_worker_index()
    worker_slot = idx % max(1, int(search_config.n_jobs))
    node_sets: list[list[int]] = []
    if mode in {"numa_spread", "numa_partition", "single_core_smt"}:
        node_sets = _numa_node_cpu_sets()
    node_cpus: set[int] | None = None
    if node_sets:
        node_cpus = set(node_sets[worker_slot % len(node_sets)])

    affinity: set[int] | None = None
    if mode == "single_core_smt":
        pairs = _thread_sibling_pairs()
        if pairs:
            if node_cpus is not None:
                filtered = [pair for pair in pairs if pair[0] in node_cpus and pair[1] in node_cpus]
            else:
                filtered = pairs
            if filtered:
                local_idx = idx
                if node_sets:
                    local_idx = worker_slot // len(node_sets)
                affinity = set(filtered[local_idx % len(filtered)])
    elif mode == "numa_spread" and node_cpus is not None:
        affinity = node_cpus
    elif mode == "numa_partition":
        pairs = _thread_sibling_pairs()
        if pairs and node_sets:
            n_nodes = len(node_sets)
            node_id = worker_slot % n_nodes
            local_idx = worker_slot // n_nodes
            workers_per_node = max(1, int(math.ceil(int(search_config.n_jobs) / float(n_nodes))))
            node_cpus_set = set(node_sets[node_id])
            node_pairs = [pair for pair in pairs if pair[0] in node_cpus_set and pair[1] in node_cpus_set]
            reserve = max(0, int(search_config.reserve_physical_cores_per_numa_node))
            usable_pairs = node_pairs[: max(0, len(node_pairs) - reserve)]
            if usable_pairs:
                base = len(usable_pairs) // workers_per_node
                extra = len(usable_pairs) % workers_per_node
                # contiguous split for better cache locality
                start = local_idx * base + min(local_idx, extra)
                length = base + (1 if local_idx < extra else 0)
                if length > 0:
                    chunk = usable_pairs[start : start + length]
                    affinity = {cpu for pair in chunk for cpu in pair}
            if not affinity and usable_pairs:
                affinity = set(usable_pairs[local_idx % len(usable_pairs)])

    if not affinity:
        return

    try:
        os.sched_setaffinity(0, affinity)
        pid = os.getpid()
        if pid not in _AFFINITY_LOGGED:
            _AFFINITY_LOGGED.add(pid)
            node_label = "n/a"
            if node_sets:
                node_label = str(worker_slot % len(node_sets))
            print(
                f"[parallel-search] worker_affinity pid={pid} mode={mode} node={node_label} "
                f"cpus={sorted(affinity)} omp={os.environ.get('OMP_NUM_THREADS', '?')}",
                flush=True,
            )
    except Exception:
        return


def _configure_blas_threads(config: ParallelSearchConfig) -> None:
    """Configure per-process BLAS threads before parallel worker spawn."""
    per_chain = max(1, int(config.cores_per_chain))
    explicit = config.blas_threads_per_worker
    if explicit is not None:
        per_process_threads = max(1, int(explicit))
    else:
        n_jobs = max(1, int(config.n_jobs))
        chains = max(1, int(config.chains))
        nuts_backend = str(config.nuts_backend).strip().lower()
        # JAX backends: one process per joblib worker (chains via pmap inside).
        if nuts_backend in {"numpyro", "blackjax"}:
            concurrent_workers = n_jobs
        else:
            concurrent_workers = n_jobs * chains
        per_process_threads = max(1, int(config.blas_total_cores) // concurrent_workers)
    per_process_threads = min(per_process_threads, per_chain)
    for env_key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_key] = str(per_process_threads)


def _clear_pytensor_cache() -> None:
    """Best-effort PyTensor cache cleanup."""
    try:
        from pytensor.link.c.basic import get_module_cache

        cache = get_module_cache()
        if hasattr(cache, "clear"):
            cache.clear()
    except Exception:
        pass
    try:
        import pytensor

        if hasattr(pytensor, "gof") and hasattr(pytensor.gof, "cc"):
            cache_obj = pytensor.gof.cc.get_module_cache()
            if hasattr(cache_obj, "clear"):
                cache_obj.clear()
    except Exception:
        pass


def _format_features(feature_selection: dict[str, list[str]]) -> str:
    """Format feature selection for CSV output."""
    parts = []
    for group_name in sorted(feature_selection):
        metrics = ", ".join(sorted(feature_selection[group_name]))
        parts.append(f"{group_name}: {metrics}")
    return "; ".join(parts)


def _format_likelihoods(parameter_selection: dict[str, dict[str, Any]]) -> str:
    """Format likelihood selection for CSV output."""
    parts = []
    for metric_name in sorted(parameter_selection):
        spec = parameter_selection.get(metric_name) or {}
        parts.append(f"{metric_name}={spec.get('likelihood', '?')}")
    return "; ".join(parts)


def verify_parallel_search_helpers() -> None:
    """Fast deterministic checks for config-grid helpers."""
    feature_configs = _generate_feature_configs(max_features=3, feature_groups=("concat", "even", "odd"))
    assert len(feature_configs) == 41, f"Expected 41 feature configs, got {len(feature_configs)}"

    test_selection = {"concat": ["mean", "range"]}
    combos = _generate_likelihood_combos(
        test_selection,
        mean_likelihoods=["student_t", "lognormal"],
        range_likelihoods=["beta", "lognormal", "interval_inflated_beta"],
        shape_shift_likelihoods=["lognormal", "gamma"],
    )
    assert len(combos) == 6, f"Expected 6 likelihood combos, got {len(combos)}"
    assert all("mean" in c and "range" in c for c in combos), "Missing metric in likelihood combo"


def verify_resume_skip(
    configs: list[dict[str, Any]],
    completed_fingerprints: set[str],
) -> tuple[int, int]:
    """Return (total, to_run) counts for resume logic sanity checks."""
    total = len(configs)
    to_run = 0
    for cfg in configs:
        fp = changepoint_model_config_fingerprint(cfg)
        if fp not in completed_fingerprints:
            to_run += 1
    return total, to_run


def verify_minimal_fit_smoke(
    export_base_cfg: dict[str, Any],
    *,
    out_dir: str,
) -> dict[str, Any]:
    """Minimal one-model smoke check for _fit_one_model pipeline."""
    cfg = ParallelSearchConfig(
        n_points_choices=[12],
        overlap_choices=[0.0],
        max_features=1,
        draws=50,
        tune=50,
        chains=2,
        n_jobs=1,
        gc_frequency=10,
        out_dir=Path(out_dir),
    )
    feature_cfg = {"concat": ["mean"]}
    param_cfg = {"mean": {"likelihood": "student_t"}}
    rem_params = validate_rem_profile_params(
        {"n_points_per_day": 12, "overlap": 0.0, "rem_stage": cfg.rem_stage}
    )

    profile_dir = Path(out_dir) / "smoke_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    export_cfg = dict(export_base_cfg)
    export_cfg.update(
        {
            "output_dir": str(profile_dir),
            "n_points_per_day": int(rem_params["n_points_per_day"]),
            "overlap": float(rem_params["overlap"]),
            "rem_stage": int(rem_params["rem_stage"]),
            "window_days": int(cfg.window_days),
        }
    )
    export_result = export_rem_profiles_10days_cached_only(**export_cfg)
    prep = prepare_model_data(csv_path=export_result["paths"]["nanpad_output_csv"])
    n_chunks = expected_fixed_n_chunk_count(
        n_points_per_day=int(rem_params["n_points_per_day"]),
        n_days=int(cfg.window_days),
    )
    model_cfg = {
        "rem_profile_params": dict(rem_params),
        "n_chunks": int(n_chunks),
        "feature_selection": feature_cfg,
        "parameter_selection": param_cfg,
        "tau_threshold": 5.0,
    }
    return _fit_one_model(
        config=model_cfg,
        data_norm=np.asarray(prep["data_norm"]),
        data_raw=np.asarray(prep["data_raw"]),
        good_indices=np.asarray(prep["good_indices"], dtype=int),
        n_points=int(rem_params["n_points_per_day"]),
        search_config=cfg,
    )


def default_profile_grid() -> list[dict[str, int | float]]:
    """Expose the default 9-profile REM grid."""
    return [dict(x) for x in REM_PROFILE_CHOICES]


__all__ = [
    "ParallelSearchConfig",
    "compare_scoring_records",
    "default_profile_grid",
    "run_parallel_search",
    "verify_minimal_fit_smoke",
    "verify_parallel_search_helpers",
    "verify_resume_skip",
    "_generate_feature_configs",
    "_generate_likelihood_combos",
]
