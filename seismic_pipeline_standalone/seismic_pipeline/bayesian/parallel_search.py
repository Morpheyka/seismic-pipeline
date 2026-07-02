"""Parallel exhaustive search with profile-level caching and memory management."""
from __future__ import annotations

import gc
import json
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
from seismic_pipeline.bayesian.search_common import build_summary_var_names
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


@dataclass
class ParallelSearchConfig:
    """Configuration for parallel exhaustive search."""

    # REM profile grid.
    n_points_choices: list[int] = field(default_factory=lambda: [12, 24, 48])
    overlap_choices: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5])

    # Feature grid (max 3 features per config).
    max_features: int = 3

    # Likelihood families.
    mean_likelihoods: list[str] = field(
        default_factory=lambda: list(LIKELIHOOD_CHOICES_BY_METRIC["mean"])
    )
    range_likelihoods: list[str] = field(
        default_factory=lambda: list(LIKELIHOOD_CHOICES_BY_METRIC["range"])
    )

    # MCMC settings.
    draws: int = 500
    tune: int = 1000
    chains: int = 4
    cores_per_chain: int = 1
    tau_mode: str = "marginalized"
    tau_lower: int = 3
    tau_upper: int = 8
    tau_threshold: float = 5.0

    # Parallelism.
    n_jobs: int = 4

    # Memory.
    gc_frequency: int = 10
    blas_total_cores: int = 10

    # Pareto-k.
    pareto_threshold: float = 0.7
    record_pareto_events: bool = True

    # Paths.
    out_dir: Path = Path("./run_output_8day_parallel")

    # Data export.
    window_days: int = 8
    rem_stage: int = 2

    # Convergence filters.
    rhat_threshold: float = 1.05
    ess_threshold: float = 100.0

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

    if "events" not in export_base_cfg:
        raise ValueError("export_base_cfg must include 'events' for REM export.")

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
    for profile_idx, rem_params in enumerate(profile_grid, start=1):
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
        prep = prepare_model_data(csv_path=export_result["paths"]["nanpad_output_csv"])
        data_norm = np.asarray(prep["data_norm"], dtype=float)
        data_raw = np.asarray(prep["data_raw"], dtype=float)
        good_indices = np.asarray(prep["good_indices"], dtype=int)
        n_chunks = expected_fixed_n_chunk_count(
            n_points_per_day=n_points,
            n_days=int(config.window_days),
        )

        feature_configs = _generate_feature_configs(max_features=config.max_features)
        model_configs: list[dict[str, Any]] = []
        for feat_cfg in feature_configs:
            for param_sel in _generate_likelihood_combos(
                feature_selection=feat_cfg,
                mean_likelihoods=config.mean_likelihoods,
                range_likelihoods=config.range_likelihoods,
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
            verbose=verbose,
        )
        all_results.extend(profile_results)
        completed_fingerprints.update(
            str(rec.get("fingerprint")) for rec in profile_results if rec.get("fingerprint")
        )

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
        results_df = results_df.assign(_sort_loo=sort_key)
        results_df = results_df.sort_values("_sort_loo", ascending=False).drop(columns=["_sort_loo"])
        results_df = results_df.reset_index(drop=True)
        results_df["rank_by_loo"] = np.arange(1, len(results_df) + 1, dtype=int)

    output_csv = out_dir / "exhaustive_search_parallel.csv"
    results_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"[parallel-search] saved: {output_csv}", flush=True)
    return results_df


def _generate_feature_configs(max_features: int = 3) -> list[dict[str, list[str]]]:
    """Generate all non-empty feature combinations up to max_features blocks."""
    if max_features < 1:
        raise ValueError("max_features must be >= 1")
    all_features = [
        ("concat", "mean"),
        ("concat", "range"),
        ("even", "mean"),
        ("even", "range"),
        ("odd", "mean"),
        ("odd", "range"),
    ]
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
        else:
            raise ValueError(f"Unsupported metric '{metric}' in feature_selection.")
        opts = sorted({x for x in opts if x})
        if not opts:
            raise ValueError(f"No likelihood options provided for metric '{metric}'.")
        lik_by_metric[metric] = opts

    combos: list[dict[str, dict[str, str]]] = []
    combo_values = [lik_by_metric[m] for m in active_metrics]
    for values in product(*combo_values):
        combos.append(
            {
                metric: {"likelihood": likelihood}
                for metric, likelihood in zip(active_metrics, values)
            }
        )
    return combos


def _fit_models_parallel(
    configs: list[dict[str, Any]],
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    good_indices: np.ndarray,
    n_points: int,
    search_config: ParallelSearchConfig,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Fit models in parallel by batches of search_config.n_jobs."""
    if not configs:
        return []
    n_total = len(configs)
    results: list[dict[str, Any]] = []
    completed = 0
    next_gc_mark = int(max(1, search_config.gc_frequency))
    batch_size = max(1, int(search_config.n_jobs))

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
            )
            for cfg in batch_configs
        )
        results.extend(batch_results)
        completed += len(batch_configs)
        while completed >= next_gc_mark:
            gc.collect()
            _clear_pytensor_cache()
            if verbose:
                print(
                    f"[parallel-search] GC cleanup at {completed}/{n_total} models",
                    flush=True,
                )
            next_gc_mark += int(max(1, search_config.gc_frequency))
    return results


def _fit_one_model(
    *,
    config: dict[str, Any],
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    good_indices: np.ndarray,
    n_points: int,
    search_config: ParallelSearchConfig,
) -> dict[str, Any]:
    """Fit a single config, record diagnostics and Pareto influence (no refits)."""
    fingerprint = changepoint_model_config_fingerprint(config)
    try:
        n_chunks = int(config["n_chunks"])
        feature_selection = config["feature_selection"]
        parameter_selection = config["parameter_selection"]
        group_data = build_group_data(
            data_norm,
            n_chunks=n_chunks,
            feature_selection=feature_selection,
            data_raw=data_raw,
            window_days=int(search_config.window_days),
            n_points_per_day=int(n_points),
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
        trace = sample_model(
            model,
            draws=int(search_config.draws),
            tune=int(search_config.tune),
            nuts_backend="pymc",
            chains=int(search_config.chains),
            cores=int(search_config.cores_per_chain),
            progressbar=False,
        )

        summary_vars = build_summary_var_names(group_data, trace)
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
        loo_k_max, loo_n_over, influential_event_indices, _ = collect_pareto_k_stats(
            trace,
            model,
            pareto_threshold=float(search_config.pareto_threshold),
        )
        pareto_k_values = _pareto_k_values(trace=trace, model=model)

        n_active_features = int(sum(len(vals or []) for vals in feature_selection.values()))
        n_model_events = int(data_norm.shape[0])
        elpd_loo = float(score_parts.get("elpd_loo", float("nan")))
        if n_model_events > 0:
            elpd_loo_per_event = elpd_loo / float(n_model_events)
        else:
            elpd_loo_per_event = float("nan")
        if n_active_features > 0 and n_model_events > 0:
            elpd_loo_per_feature_event = elpd_loo / float(n_active_features * n_model_events)
        else:
            elpd_loo_per_feature_event = float("nan")

        record = {
            "fingerprint": fingerprint,
            "status": "ok",
            "error": None,
            "elpd_loo": elpd_loo,
            "loo": float(score_parts.get("loo", float("nan"))),
            "elpd_loo_per_event": elpd_loo_per_event,
            "elpd_loo_per_feature": elpd_loo_per_feature_event,
            "elpd_loo_per_feature_event": elpd_loo_per_feature_event,
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
            "pareto_k_values": (
                json.dumps(pareto_k_values)
                if search_config.record_pareto_events
                else "[]"
            ),
            "influential_event_indices": (
                json.dumps([int(x) for x in influential_event_indices])
                if search_config.record_pareto_events
                else "[]"
            ),
            "good_indices": json.dumps(good_indices.tolist()),
            "n_active_features": n_active_features,
            "n_model_events": n_model_events,
            "features": _format_features(feature_selection),
            "likelihoods": _format_likelihoods(parameter_selection),
            "n_points": int(config["rem_profile_params"]["n_points_per_day"]),
            "overlap": float(config["rem_profile_params"]["overlap"]),
            "config_json": json.dumps(config, sort_keys=True, default=str),
        }
        del trace
        del model
        gc.collect()
        return record
    except Exception as exc:
        traceback.print_exc()
        warnings.warn(f"Model {fingerprint} failed: {exc}")
        gc.collect()
        return {
            "fingerprint": fingerprint,
            "status": "failed",
            "error": str(exc),
            "config_json": json.dumps(config, sort_keys=True, default=str),
            "elpd_loo_per_feature_event": float("nan"),
        }


def _pareto_k_values(trace, model) -> list[float]:
    """Return pointwise Pareto-k values from LOO (empty on failure)."""
    try:
        idata = idata_for_waic_from_trace(trace, model)
        loo_obj = az.loo(idata, scale="log", pointwise=True)
        pareto = getattr(loo_obj, "pareto_k", None)
        if pareto is None:
            return []
        arr = np.asarray(pareto, dtype=float).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return []


def _configure_blas_threads(config: ParallelSearchConfig) -> None:
    """Configure per-process BLAS threads before parallel worker spawn."""
    n_jobs = max(1, int(config.n_jobs))
    per_process_threads = max(1, int(config.blas_total_cores) // n_jobs)
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
    feature_configs = _generate_feature_configs(max_features=3)
    assert len(feature_configs) == 41, f"Expected 41 feature configs, got {len(feature_configs)}"

    test_selection = {"concat": ["mean", "range"]}
    combos = _generate_likelihood_combos(
        test_selection,
        mean_likelihoods=["student_t", "lognormal"],
        range_likelihoods=["beta", "lognormal", "interval_inflated_beta"],
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
    "default_profile_grid",
    "run_parallel_search",
    "verify_minimal_fit_smoke",
    "verify_parallel_search_helpers",
    "verify_resume_skip",
    "_generate_feature_configs",
    "_generate_likelihood_combos",
]
