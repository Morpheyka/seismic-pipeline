"""Auto-split from rem_profiles_export_10days_lib.py."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

from seismic_pipeline.features.runtime import (
    get_runtime_data_raw,
    get_runtime_export_cfg,
    get_runtime_prepare_cfg,
)
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    prepare_model_data,
    precompute_all_features,
    set_runtime_data_norm,
    group_data_from_precomputed,
    _parse_feature_selection,
)
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only
from seismic_pipeline.config.changepoint_defaults import validate_rem_profile_params
from seismic_pipeline.bayesian.diagnostics import (
    collect_pareto_k_stats,
    score_changepoint_trace,
    _available_varnames,
    changepoint_model_config_fingerprint,
)
from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.priors import _default_parameter_selection
from seismic_pipeline.config.changepoint_defaults import LIKELIHOOD_CHOICES_BY_METRIC
from seismic_pipeline.bayesian.search_common import (
    _clone_config,
    _fit_config_once,
    _validate_feature_selection_for_n_chunks,
    _build_summary_var_names,
)

def propose_changepoint_model_config(
    current: dict,
    proposal_options: dict,
    rng: np.random.Generator,
) -> dict:
    """One symmetric random-walk proposal on the discrete model space.

    proposal_options keys:
    - rem_profile_choices: list[dict] with n_points_per_day, overlap, rem_stage
    - n_chunks_choices: list[int] (even if concat group may be used)
    - allowed_groups: list[str] subset of concat, odd, even, all
    - allowed_metrics: list[str]
    - likelihood_choices_by_metric: dict[metric, list[str]] optional; defaults used if missing

    Move types (uniform over 6): REM profile, ``n_chunks``, feature toggle, likelihood family,
    **swap** metric within a group (fixed model size), **perturb** a prior scale (symmetric
    multiplicative random walk on ``log sigma``).
    """
    prop = _clone_config(current)
    rem_choices = list(proposal_options.get("rem_profile_choices") or [])
    n_choices = list(proposal_options.get("n_chunks_choices") or [])
    groups = list(proposal_options.get("allowed_groups") or ["concat", "odd", "even"])
    metrics = list(proposal_options.get("allowed_metrics") or ["mean", "range"])
    like_map: dict[str, list[str]] = dict(proposal_options.get("likelihood_choices_by_metric") or {})
    default_likes = sorted(
        {c for choices in LIKELIHOOD_CHOICES_BY_METRIC.values() for c in choices}
    )

    move = int(rng.integers(0, 6))
    n_chunks = int(prop["n_chunks"])

    if move == 0 and len(rem_choices) > 1:
        cur_rem = prop.get("rem_profile_params") or {}
        others = [r for r in rem_choices if r != cur_rem]
        if others:
            prop["rem_profile_params"] = dict(rng.choice(others))
        return prop

    if move == 1 and len(n_choices) > 1:
        others = [n for n in n_choices if int(n) != n_chunks]
        if others:
            prop["n_chunks"] = int(rng.choice(others))
        return prop

    if move == 2:
        fs = dict(prop.get("feature_selection") or {})
        pair_pool: List[Tuple[str, str]] = []
        for g in groups:
            for m in metrics:
                pair_pool.append((g, m))
        if not pair_pool:
            return prop
        g, m = pair_pool[int(rng.integers(0, len(pair_pool)))]
        if g not in fs:
            fs[g] = []
        lst = list(fs[g])
        if m in lst:
            lst = [x for x in lst if x != m]
            if lst:
                fs[g] = lst
            else:
                del fs[g]
        else:
            lst.append(m)
            fs[g] = sorted(set(lst), key=lambda x: metrics.index(x) if x in metrics else 0)
        if fs and _validate_feature_selection_for_n_chunks(fs, int(prop["n_chunks"])):
            prop["feature_selection"] = fs
        return prop

    if move == 3:
        fs = dict(prop.get("feature_selection") or {})
        ps = dict(prop.get("parameter_selection") or {})
        active: List[str] = []
        for feats in fs.values():
            for feat in feats:
                if feat not in active:
                    active.append(feat)
        if not active:
            return prop
        feat = str(rng.choice(active))
        cur_like = str(
            ps.get(feat, {}).get("likelihood", _default_parameter_selection()[feat].get("likelihood", "normal"))
        ).lower()
        if feat in like_map:
            choices = [str(c).lower() for c in like_map[feat]]
        elif feat in LIKELIHOOD_CHOICES_BY_METRIC:
            choices = [str(c).lower() for c in LIKELIHOOD_CHOICES_BY_METRIC[feat]]
        else:
            choices = [c.lower() for c in default_likes]
        alts = [c for c in choices if c != cur_like]
        if not alts:
            return prop
        new_like = str(rng.choice(alts))
        if feat not in ps:
            ps[feat] = {}
        ps[feat] = dict(ps[feat])
        ps[feat]["likelihood"] = new_like
        prop["parameter_selection"] = ps
        return prop

    if move == 4:
        # Swap: replace one active metric in a group with a different allowed metric (same group).
        fs = dict(prop.get("feature_selection") or {})
        nonempty_groups = [g for g in groups if g in fs and fs[g]]
        if not nonempty_groups:
            return prop
        g = str(rng.choice(nonempty_groups))
        cur_metrics = list(fs[g])
        if not cur_metrics:
            return prop
        m_old = str(rng.choice(cur_metrics))
        alternatives = [m for m in metrics if m != m_old]
        if not alternatives:
            return prop
        m_new = str(rng.choice(alternatives))
        new_lst = [m for m in cur_metrics if m != m_old]
        if m_new not in new_lst:
            new_lst.append(m_new)
        fs[g] = sorted(set(new_lst), key=lambda x: metrics.index(x) if x in metrics else 0)
        if fs and _validate_feature_selection_for_n_chunks(fs, int(prop["n_chunks"])):
            prop["feature_selection"] = fs
        return prop

    if move == 5:
        # Perturb: symmetric random walk on log(scale) for ``mu_prior.sigma`` or ``sigma_prior.sigma``.
        fs = dict(prop.get("feature_selection") or {})
        ps = dict(prop.get("parameter_selection") or {})
        active = [f for feats in fs.values() for f in feats]
        active = list(dict.fromkeys(active))
        if not active:
            return prop
        feat = str(rng.choice(active))
        if feat not in ps:
            ps[feat] = {}
        ps[feat] = dict(ps[feat])
        key = str(rng.choice(["sigma_prior", "mu_prior"]))
        if key not in ps[feat]:
            ps[feat][key] = (
                {"dist": "halfnormal", "sigma": 1.0}
                if key == "sigma_prior"
                else {"dist": "normal", "mu": 0.0, "sigma": 1.0}
            )
        ps[feat][key] = dict(ps[feat][key])
        spin = float(np.exp(rng.uniform(-0.45, 0.45)))
        if key == "sigma_prior" and str(ps[feat][key].get("dist", "")).lower() in {"halfnormal", "halfstudentt"}:
            base = float(ps[feat][key].get("sigma", 1.0))
            ps[feat][key]["sigma"] = max(0.05, base * spin)
        elif key == "mu_prior" and str(ps[feat][key].get("dist", "")).lower() == "normal":
            base = float(ps[feat][key].get("sigma", 1.0))
            ps[feat][key]["sigma"] = max(0.05, base * spin)
        prop["parameter_selection"] = ps
        return prop

    return prop
def _nonempty_subsets(items: list[tuple[str, str]]) -> Iterable[tuple[tuple[str, str], ...]]:
    for r in range(1, len(items) + 1):
        yield from itertools.combinations(items, r)
def _group_metric_shape_signature_from_config(
    config: dict,
    *,
    n_events: int,
) -> tuple[tuple[str, str, int, int], ...]:
    n_chunks = int(config["n_chunks"])
    idx_map = _chunk_group_idx_map(n_chunks)
    fs = _parse_feature_selection(config.get("feature_selection", {}))
    blocks: list[tuple[str, str, int, int]] = []
    for group_name in sorted(fs.keys()):
        chunk_count = int(len(idx_map[group_name]))
        metrics = sorted(set(str(m).strip().lower() for m in fs[group_name]))
        for metric_name in metrics:
            blocks.append((group_name, metric_name, int(n_events), chunk_count))
    return tuple(sorted(blocks))
def _exhaustive_model_signature(
    config: dict,
    *,
    include_rem_profile: bool = False,
) -> tuple:
    """Signature of what exhaustive_model_search actually fits on fixed data_norm."""
    n_chunks = int(config.get("n_chunks"))
    tau_threshold = float(config.get("tau_threshold", 6.0))

    fs_raw = config.get("feature_selection") or {}
    fs_sig: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
        sorted(
            (
                str(group_name).strip().lower(),
                tuple(
                    sorted(
                        str(metric_name).strip().lower()
                        for metric_name in (metrics or [])
                    )
                ),
            )
            for group_name, metrics in fs_raw.items()
        )
    )

    ps_raw = config.get("parameter_selection") or {}
    ps_sig: tuple[tuple[str, str], ...] = tuple(
        sorted(
            (
                str(metric_name).strip().lower(),
                str((params or {}).get("likelihood", "")).strip().lower(),
            )
            for metric_name, params in ps_raw.items()
        )
    )

    rem_sig: tuple[int, float, int] | None = None
    if include_rem_profile:
        rem = config.get("rem_profile_params") or {}
        if "n_points_per_day" in rem:
            rem_sig = (
                int(rem.get("n_points_per_day", 0)),
                float(rem.get("overlap", 0.0)),
                int(rem.get("rem_stage", 0)),
            )
        else:
            rem_sig = (
                int(rem.get("window_size_hours", 0)),
                int(rem.get("step_size_hours", 0)),
                int(rem.get("rem_stage", 0)),
            )

    # By default rem_profile_params are excluded because exhaustive_model_search
    # can run on fixed `data_norm`. When runtime export/prepare configs exist and
    # per-REM recalculation is enabled, rem_sig is included.
    return (n_chunks, tau_threshold, fs_sig, ps_sig, rem_sig)
def _generate_exhaustive_configs(
    proposal_options: dict,
    *,
    tau_threshold: float | None = None,
) -> list[dict]:
    if tau_threshold is None:
        tau_threshold = float(proposal_options.get("tau_threshold", 6.0))
    rem_profile_choices = list(proposal_options.get("rem_profile_choices") or [])
    n_chunks_choices = [int(x) for x in (proposal_options.get("n_chunks_choices") or [])]
    allowed_groups = [str(g).strip().lower() for g in (proposal_options.get("allowed_groups") or [])]
    allowed_metrics = [str(m).strip().lower() for m in (proposal_options.get("allowed_metrics") or [])]
    likelihood_choices_by_metric = dict(proposal_options.get("likelihood_choices_by_metric") or {})

    if not rem_profile_choices:
        raise ValueError("proposal_options['rem_profile_choices'] must be a non-empty list.")
    if not n_chunks_choices:
        raise ValueError("proposal_options['n_chunks_choices'] must be a non-empty list.")
    if not allowed_groups:
        raise ValueError("proposal_options['allowed_groups'] must be a non-empty list.")
    if not allowed_metrics:
        raise ValueError("proposal_options['allowed_metrics'] must be a non-empty list.")

    defaults = _default_parameter_selection()
    valid_groups = {"all", "odd", "even", "concat"}
    valid_metrics = set(defaults.keys())
    unknown_groups = sorted(set(allowed_groups) - valid_groups)
    unknown_metrics = sorted(set(allowed_metrics) - valid_metrics)
    if unknown_groups:
        raise ValueError(f"Unknown groups in allowed_groups: {unknown_groups}")
    if unknown_metrics:
        raise ValueError(f"Unknown metrics in allowed_metrics: {unknown_metrics}")

    rem_normed: list[dict] = []
    for rem in rem_profile_choices:
        if not isinstance(rem, dict):
            raise ValueError("Each rem_profile choice must be a dict.")
        rem_normed.append(validate_rem_profile_params(rem))

    block_space = sorted(
        (g, m)
        for g in allowed_groups
        for m in allowed_metrics
        if m != "shape_shift" or g in {"concat", "odd", "even", "all"}
    )
    out: list[dict] = []
    for rem_params in rem_normed:
        for n_chunks in n_chunks_choices:
            if int(n_chunks) <= 0:
                raise ValueError(f"n_chunks must be > 0, got {n_chunks}")
            for subset in _nonempty_subsets(block_space):
                feature_selection: dict[str, list[str]] = {}
                for group_name, metric_name in subset:
                    feature_selection.setdefault(group_name, [])
                    if metric_name not in feature_selection[group_name]:
                        feature_selection[group_name].append(metric_name)
                feature_selection = {
                    g: sorted(v)
                    for g, v in sorted(feature_selection.items())
                }
                active_metrics = sorted({m for _, m in subset})
                metric_likelihood_choices: list[list[str]] = []
                for metric_name in active_metrics:
                    opts_raw = likelihood_choices_by_metric.get(metric_name)
                    if opts_raw is None:
                        opts = [str(defaults[metric_name]["likelihood"]).strip().lower()]
                    else:
                        opts = [str(x).strip().lower() for x in list(opts_raw)]
                        opts = [x for x in opts if x]
                        if not opts:
                            raise ValueError(
                                "likelihood_choices_by_metric contains an empty list "
                                f"for metric '{metric_name}'."
                            )
                    metric_likelihood_choices.append(sorted(set(opts)))
                for likelihood_combo in itertools.product(*metric_likelihood_choices):
                    parameter_selection = {
                        metric_name: {"likelihood": likelihood_name}
                        for metric_name, likelihood_name in zip(active_metrics, likelihood_combo)
                    }
                    out.append(
                        {
                            "rem_profile_params": dict(rem_params),
                            "n_chunks": int(n_chunks),
                            "feature_selection": copy.deepcopy(feature_selection),
                            "parameter_selection": parameter_selection,
                            "tau_threshold": float(tau_threshold),
                        }
                    )
    return out
def exhaustive_model_search(
    proposal_options: dict,
    data_norm: np.ndarray,
    *,
    configs: list[dict] | None = None,
    draws: int = 500,
    tune: int = 1000,
    nuts_backend: str = "pymc",
    chains: int = 2,
    cores: int | None = None,
    tau_mode: str = "marginalized",
    tau_lower: int = 2,
    tau_upper: int = 10,
    cache_fits: bool = True,
    seed: int = 42,
    verbose: bool = True,
    progressbar: bool = True,
    progress_desc: str = "exhaustive models",
    progress_position: int | None = None,
) -> dict:
    """Evaluate all changepoint model configurations from a proposal grid.

    Each result record's ``loo`` field (and ``elpd_loo`` when present) uses ArviZ
    ``elpd_loo`` (log predictive density sum), not LOO-IC (``-2 * elpd_loo``).
    """
    t0 = time.perf_counter()
    _ = np.random.default_rng(int(seed))
    runtime_export_cfg_base = dict(get_runtime_export_cfg()) if get_runtime_export_cfg() is not None else None
    base_export_output_dir = (
        str(runtime_export_cfg_base.get("output_dir", "."))
        if runtime_export_cfg_base is not None
        else None
    )
    tau_threshold = float(proposal_options.get("tau_threshold", 6.0))
    pareto_threshold = float(proposal_options.get("pareto_threshold", 0.7))
    max_pareto_retries = int(proposal_options.get("max_pareto_retries", 3))
    enable_pareto_refit_raw = proposal_options.get("enable_pareto_refit", True)
    if isinstance(enable_pareto_refit_raw, str):
        enable_pareto_refit = enable_pareto_refit_raw.strip().lower() not in {"0", "false", "no", "off"}
    else:
        enable_pareto_refit = bool(enable_pareto_refit_raw)

    use_rem_profile_recalc = False
    if configs is not None:
        all_configs = [_clone_config(cfg) for cfg in configs]
        n_total = len(all_configs)
        unique_configs = all_configs
        n_filtered_degenerate = 0
    else:
        all_configs = _generate_exhaustive_configs(proposal_options, tau_threshold=tau_threshold)
        n_total = len(all_configs)
        n_events = int(np.asarray(data_norm).shape[0])
        use_rem_profile_recalc = (
            get_runtime_export_cfg() is not None
            and any(cfg.get("rem_profile_params") for cfg in all_configs)
        )

        dedup_seen: set[tuple] = set()
        unique_configs = []
        n_filtered_degenerate = 0
        for cfg in all_configs:
            # Deduplicate by model semantics actually used by
            # exhaustive_model_search on fixed `data_norm`.
            # Keep likelihood differences; collapse rem_profile-only duplicates.
            sig = _exhaustive_model_signature(
                cfg,
                include_rem_profile=use_rem_profile_recalc,
            )
            if sig in dedup_seen:
                n_filtered_degenerate += 1
                continue
            dedup_seen.add(sig)
            unique_configs.append(cfg)

    trace_cache: dict[str, Any] = {}
    model_cache: dict[str, Any] = {}
    score_cache: dict[str, dict[str, Any]] = {}
    rem_data_cache: dict[str, dict[str, np.ndarray]] = {}
    results: list[dict[str, Any]] = []
    n_fit_errors = 0
    n_pareto_skipped = 0

    def rem_output_dir_for_config(cfg: dict) -> str | None:
        rpp = cfg.get("rem_profile_params")
        if not rpp or runtime_export_cfg_base is None or base_export_output_dir is None:
            return None
        return os.path.join(
            base_export_output_dir,
            (
                f"rem_n{int(rpp['n_points_per_day'])}"
                f"_ov{float(rpp['overlap']):.2f}"
                f"_stage{int(rpp['rem_stage'])}"
            ),
        )

    def ensure_data_for_config(cfg: dict) -> tuple[np.ndarray, np.ndarray | None]:
        if not use_rem_profile_recalc:
            raw = get_runtime_data_raw()
            return data_norm, raw
        rpp = cfg.get("rem_profile_params")
        if not rpp or runtime_export_cfg_base is None:
            return data_norm, get_runtime_data_raw()
        rem_key = json.dumps(
            {
                "n_points_per_day": int(rpp["n_points_per_day"]),
                "overlap": float(rpp["overlap"]),
                "rem_stage": int(rpp["rem_stage"]),
            },
            sort_keys=True,
        )
        if rem_key in rem_data_cache:
            cached = rem_data_cache[rem_key]
            return cached["data_norm"], cached.get("data_raw")
        export_cfg = dict(runtime_export_cfg_base)
        rem_output_dir = rem_output_dir_for_config(cfg)
        export_cfg.update(
            {
                "output_dir": rem_output_dir,
                "n_points_per_day": int(rpp["n_points_per_day"]),
                "overlap": float(rpp["overlap"]),
                "rem_stage": int(rpp["rem_stage"]),
            }
        )
        export_result = export_rem_profiles_10days_cached_only(**export_cfg)
        prep_cfg = get_runtime_prepare_cfg() or {}
        prep = prepare_model_data(
            csv_path=export_result["paths"]["nanpad_output_csv"],
            bad_sample_indices=prep_cfg.get("bad_sample_indices"),
        )
        rem_data_cache[rem_key] = {
            "data_norm": prep["data_norm"],
            "data_raw": prep["data_raw"],
        }
        return rem_data_cache[rem_key]["data_norm"], rem_data_cache[rem_key]["data_raw"]

    iterator: Iterable[tuple[int, dict]]
    iterator = enumerate(unique_configs, start=1)
    if progressbar:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(
                iterator,
                total=len(unique_configs),
                desc=str(progress_desc),
                position=progress_position,
                leave=True,
            )
        except Exception:
            pass

    for idx, config in iterator:
        fp = changepoint_model_config_fingerprint(config)
        t_model0 = time.perf_counter()
        data_work_base, data_raw_base = ensure_data_for_config(config)
        active_event_indices = np.arange(int(data_work_base.shape[0]), dtype=int)
        removed_event_indices: list[int] = []
        pareto_retry_count = 0
        skipped_due_to_pareto = False
        try:
            while True:
                if int(active_event_indices.size) == 0:
                    raise ValueError("No events left after Pareto-k filtering.")
                data_work = data_work_base[active_event_indices]
                data_raw_work = (
                    data_raw_base[active_event_indices]
                    if data_raw_base is not None
                    else None
                )
                window_days = None
                if runtime_export_cfg_base is not None:
                    window_days = int(runtime_export_cfg_base.get("window_days", 10))
                if int(active_event_indices.size) == int(data_work_base.shape[0]):
                    cache_key = fp
                else:
                    kept_sig = ",".join(str(int(i)) for i in active_event_indices.tolist())
                    cache_key = f"{fp}::events={kept_sig}"

                if cache_fits and cache_key in score_cache:
                    score_parts = score_cache[cache_key]
                    trace = trace_cache[cache_key]
                    model = model_cache[cache_key]
                else:
                    group_data = build_group_data(
                        data_work,
                        n_chunks=int(config["n_chunks"]),
                        feature_selection=config["feature_selection"],
                        data_raw=data_raw_work,
                        window_days=window_days,
                    )
                    tu = int(tau_upper) if tau_upper is not None else None
                    model = build_changepoint_model(
                        group_data,
                        tau_lower=int(tau_lower),
                        tau_upper=tu,
                        parameter_selection=config["parameter_selection"],
                        tau_mode=tau_mode,
                    )
                    trace = sample_model(
                        model,
                        draws=draws,
                        tune=tune,
                        nuts_backend=nuts_backend,
                        chains=chains,
                        cores=cores,
                        progressbar=False,
                    )
                    summary_vars = _build_summary_var_names(group_data, trace)
                    score_parts = score_changepoint_trace(
                        trace,
                        group_data=group_data,
                        parameter_selection=config["parameter_selection"],
                        tau_threshold=float(config.get("tau_threshold", tau_threshold)),
                        summary_var_names=summary_vars if summary_vars else None,
                        model=model,
                        criterion="loo",
                        warn_on_fallback=False,
                        loo_report="elpd",
                    )
                    if cache_fits:
                        score_cache[cache_key] = score_parts
                        trace_cache[cache_key] = trace
                        model_cache[cache_key] = model

                loo_k_max, loo_n_over, _, worst_local_idx = _collect_pareto_k_stats(
                    trace,
                    model,
                    pareto_threshold=pareto_threshold,
                )
                if (
                    enable_pareto_refit
                    and max_pareto_retries > 0
                    and
                    loo_n_over > 0
                    and math.isfinite(loo_k_max)
                    and loo_k_max > pareto_threshold
                ):
                    if (
                        pareto_retry_count >= max_pareto_retries
                        or worst_local_idx is None
                        or int(active_event_indices.size) <= 1
                    ):
                        skipped_due_to_pareto = True
                        break
                    dropped_event = int(active_event_indices[int(worst_local_idx)])
                    removed_event_indices.append(dropped_event)
                    active_event_indices = np.delete(active_event_indices, int(worst_local_idx))
                    pareto_retry_count += 1
                    if verbose:
                        print(
                            f"[exhaustive] model {idx}/{len(unique_configs)} config fp={fp} "
                            f"Pareto-k={loo_k_max:.3f} > {pareto_threshold:.3f}; "
                            f"dropped event_idx={dropped_event}, retry={pareto_retry_count}",
                            flush=True,
                        )
                    continue
                break
            if skipped_due_to_pareto:
                n_pareto_skipped += 1
                if verbose:
                    print(
                        f"[exhaustive] model {idx}/{len(unique_configs)} config fp={fp} "
                        f"skipped after {pareto_retry_count} Pareto retries; "
                        f"Pareto-k={loo_k_max:.3f} > {pareto_threshold:.3f}; "
                        f"removed_event_indices={removed_event_indices}",
                        flush=True,
                    )
                continue
            elapsed = time.perf_counter() - t_model0
            record = {
                "config": _clone_config(config),
                "fingerprint": fp,
                "data_shape": tuple(int(x) for x in np.asarray(data_work).shape),
                "data_output_dir": rem_output_dir_for_config(config) if use_rem_profile_recalc else None,
                "bad_sample_indices": list((_RUNTIME_LAST_PREPARE_CFG or {}).get("bad_sample_indices") or []),
                "waic": float(score_parts.get("waic", float("nan"))),
                "waic_warning_flag": bool(score_parts.get("waic_warning_flag", False)),
                "waic_warning_messages": list(score_parts.get("waic_warning_messages") or []),
                "loo": float(score_parts.get("loo", float("nan"))),
                "elpd_loo": float(score_parts.get("elpd_loo", float("nan"))),
                "loo_ic": float(score_parts.get("loo_ic", float("nan"))),
                "loo_pareto_k_max": loo_k_max,
                "loo_n_over_threshold": int(loo_n_over),
                "r_hat_max": float(score_parts.get("r_hat_max", float("nan"))),
                "ess_min_bulk": float(score_parts.get("ess_min_bulk", float("nan"))),
                "ess_min_tail": float(score_parts.get("ess_min_tail", float("nan"))),
                "bfmi": float(score_parts.get("bfmi", score_parts.get("bfmi_approx", float("nan")))),
                "n_divergences": int(score_parts.get("n_divergences", 0)),
                "p_tau_gt_threshold": float(score_parts.get("p_tau_gt_threshold", float("nan"))),
                "e_tau": float(score_parts.get("e_tau", float("nan"))),
                "tau_std": float(score_parts.get("tau_std", float("nan"))),
                "p_tau_gt_6": float(p_tau_gt_from_trace(trace, 6.0)),
                "tau_map": int(score_parts.get("map_tau", -1)),
                "tau_map_concentration": float(score_parts.get("tau_concentration", float("nan"))),
                "n_feature_blocks": int(score_parts.get("n_feature_blocks", 0)),
                "n_model_events": int(active_event_indices.size),
                "pareto_retry_count": int(pareto_retry_count),
                "removed_event_indices": [int(x) for x in removed_event_indices],
                "elapsed_time": float(elapsed),
                "status": "ok",
                "error": None,
            }
            n_active_features = int(
                sum(len(metrics or []) for metrics in (config.get("feature_selection") or {}).values())
            )
            n_model_events = int(active_event_indices.size)
            record["n_active_features"] = n_active_features
            if n_model_events > 0 and "elpd_loo" in record:
                record["elpd_loo_per_event"] = float(record["elpd_loo"]) / float(n_model_events)
            else:
                record["elpd_loo_per_event"] = float("nan")
            if n_active_features > 0 and n_model_events > 0 and "elpd_loo" in record:
                norm = float(record["elpd_loo"]) / float(n_active_features * n_model_events)
                record["elpd_loo_per_feature"] = norm
                record["elpd_loo_per_feature_event"] = norm
            else:
                record["elpd_loo_per_feature"] = float("nan")
                record["elpd_loo_per_feature_event"] = float("nan")
            results.append(record)
            if verbose:
                print(
                    f"[exhaustive] model {idx}/{len(unique_configs)} config fp={fp} "
                    f"elpd_loo={record['loo']:.3f} waic={record['waic']:.3f} "
                    f"r_hat={record['r_hat_max']:.3f} "
                    f"tau_map={record['tau_map']} "
                    f"P(tau)={record['tau_map_concentration']:.3f} ok",
                    flush=True,
                )
        except Exception as exc:
            n_fit_errors += 1
            elapsed = time.perf_counter() - t_model0
            err_record = {
                "config": _clone_config(config),
                "fingerprint": fp,
                "waic": float("nan"),
                "waic_warning_flag": False,
                "waic_warning_messages": [],
                "loo": float("nan"),
                "elpd_loo": float("nan"),
                "loo_ic": float("nan"),
                "loo_pareto_k_max": float("nan"),
                "loo_n_over_threshold": 0,
                "r_hat_max": float("nan"),
                "ess_min_bulk": float("nan"),
                "ess_min_tail": float("nan"),
                "bfmi": float("nan"),
                "n_divergences": 0,
                "p_tau_gt_threshold": float("nan"),
                "e_tau": float("nan"),
                "tau_std": float("nan"),
                "tau_map": -1,
                "tau_map_concentration": float("nan"),
                "n_feature_blocks": 0,
                "n_model_events": int(active_event_indices.size),
                "pareto_retry_count": int(pareto_retry_count),
                "removed_event_indices": [int(x) for x in removed_event_indices],
                "elapsed_time": float(elapsed),
                "status": "error",
                "error": str(exc),
            }
            n_active_features = int(
                sum(len(metrics or []) for metrics in (config.get("feature_selection") or {}).values())
            )
            err_record["n_active_features"] = n_active_features
            err_record["elpd_loo_per_event"] = float("nan")
            err_record["elpd_loo_per_feature"] = float("nan")
            err_record["elpd_loo_per_feature_event"] = float("nan")
            results.append(err_record)
            if verbose:
                print(
                    f"[exhaustive] model {idx}/{len(unique_configs)} config fp={fp} "
                    f"failed error={exc}",
                    flush=True,
                )

    def _loo_sort_key(rec: dict[str, Any]) -> float:
        v = float(rec.get("elpd_loo_per_feature", float("nan")))
        return v if math.isfinite(v) else float("-inf")

    results_sorted = sorted(results, key=_loo_sort_key, reverse=True)
    top_configs = [
        rec for rec in results_sorted if rec.get("status") == "ok"
    ][:20]
    n_fitted = int(sum(1 for r in results if r.get("status") == "ok"))
    n_filtered = int(n_filtered_degenerate + n_fit_errors + n_pareto_skipped)

    return {
        "results": results_sorted,
        "n_total": int(n_total),
        "n_fitted": n_fitted,
        "n_filtered": n_filtered,
        "top_configs": top_configs,
        "elapsed_total": float(time.perf_counter() - t0),
    }
def model_config_hamming_distance(config1: dict, config2: dict) -> int:
    """Number of differing parameters between two model configs."""
    d = 0
    r1 = config1.get("rem_profile_params") or {}
    r2 = config2.get("rem_profile_params") or {}
    if "n_points_per_day" in r1 or "n_points_per_day" in r2:
        rem1 = (
            int(r1.get("n_points_per_day", -1)),
            float(r1.get("overlap", -1.0)),
        )
        rem2 = (
            int(r2.get("n_points_per_day", -1)),
            float(r2.get("overlap", -1.0)),
        )
    else:
        rem1 = (
            r1.get("window_size_hours"),
            r1.get("step_size_hours"),
        )
        rem2 = (
            r2.get("window_size_hours"),
            r2.get("step_size_hours"),
        )
    if rem1 != rem2:
        d += 1
    if config1.get("n_chunks") != config2.get("n_chunks"):
        d += 1

    def _blocks(cfg: dict) -> set[tuple[str, str]]:
        fs = cfg.get("feature_selection", {})
        if not isinstance(fs, dict):
            return set()
        out: set[tuple[str, str]] = set()
        for g, ms in fs.items():
            for m in ms or []:
                out.add((str(g), str(m)))
        return out

    d += len(_blocks(config1).symmetric_difference(_blocks(config2)))
    ps1 = config1.get("parameter_selection", {})
    ps2 = config2.get("parameter_selection", {})
    all_metrics = set(ps1.keys()) | set(ps2.keys())
    for metric_name in all_metrics:
        ll1 = (ps1.get(metric_name) or {}).get("likelihood")
        ll2 = (ps2.get(metric_name) or {}).get("likelihood")
        if ll1 != ll2:
            d += 1
    return d
def compute_model_distance_matrix(configs: list[dict]) -> np.ndarray:
    """N×N Hamming distance matrix for a list of configs."""
    n = len(configs)
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = model_config_hamming_distance(configs[i], configs[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist
def _config_feature_label(config: dict) -> str:
    fs = config.get("feature_selection") or {}
    parts = []
    for group_name in sorted(fs):
        metrics = ", ".join(str(m) for m in fs[group_name])
        parts.append(f"{group_name}: {metrics}")
    return "; ".join(parts)
def exhaustive_tau_map_table(
    search_result: dict,
    *,
    top_n: int | None = 20,
    sort_by: str = "loo",
    valid_only: bool = False,
) -> pd.DataFrame:
    """Build a table of MAP tau and concentration per exhaustive-search model.

    The ``elpd_loo`` column matches exhaustive search scoring (not LOO-IC).
    """
    results = list(search_result.get("results") or [])
    if valid_only:
        results = [
            r
            for r in results
            if r.get("status") == "ok"
            and math.isfinite(float(r.get("loo", float("nan"))))
            and float(r.get("r_hat_max", float("inf"))) <= 1.05
            and float(r.get("ess_min_bulk", float("-inf"))) >= 100.0
            and int(r.get("n_divergences", 1)) == 0
        ]
    else:
        results = [r for r in results if r.get("status") == "ok"]

    key = str(sort_by).strip().lower()
    if key not in {"loo", "waic", "tau_map", "tau_map_concentration"}:
        raise ValueError("sort_by must be one of: loo, waic, tau_map, tau_map_concentration")

    def _sort_val(rec: dict) -> float:
        v = float(rec.get(key, float("nan")))
        return v if math.isfinite(v) else float("-inf")

    results = sorted(results, key=_sort_val, reverse=True)
    if top_n is not None:
        results = results[: max(1, int(top_n))]

    rows: list[dict[str, Any]] = []
    for rank, rec in enumerate(results, start=1):
        cfg = rec.get("config") or {}
        rem = cfg.get("rem_profile_params") or {}
        rows.append(
            {
                "rank": rank,
                "fingerprint": str(rec.get("fingerprint", "")),
                "features": _config_feature_label(cfg),
                "n_points": int(rem.get("n_points_per_day", rem.get("window_size_hours", -1))),
                "overlap": float(rem.get("overlap", -1.0)),
                "step_h": int(rem.get("step_size_hours", -1)),
                "elpd_loo": float(rec.get("elpd_loo", rec.get("loo", float("nan")))),
                "waic": float(rec.get("waic", float("nan"))),
                "tau_map": int(rec.get("tau_map", -1)),
                "P_tau_map": float(rec.get("tau_map_concentration", float("nan"))),
                "p_tau_gt_threshold": float(rec.get("p_tau_gt_threshold", float("nan"))),
                "r_hat_max": float(rec.get("r_hat_max", float("nan"))),
            }
        )
    return pd.DataFrame(rows)
def print_exhaustive_tau_map_table(
    search_result: dict,
    *,
    top_n: int | None = 20,
    sort_by: str = "loo",
    valid_only: bool = False,
) -> pd.DataFrame:
    """Print and return MAP-tau table for exhaustive-search models."""
    table = exhaustive_tau_map_table(
        search_result,
        top_n=top_n,
        sort_by=sort_by,
        valid_only=valid_only,
    )
    if table.empty:
        print("No exhaustive-search records with status='ok' to display.")
        return table
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return table
def _exhaustive_p_tau_gt(rec: dict, threshold: float) -> float:
    thr = float(threshold)
    if thr == 6.0 and "p_tau_gt_6" in rec:
        return float(rec.get("p_tau_gt_6", float("nan")))
    cfg_thr = float((rec.get("config") or {}).get("tau_threshold", float("nan")))
    if math.isfinite(cfg_thr) and abs(cfg_thr - thr) < 1e-9:
        return float(rec.get("p_tau_gt_threshold", float("nan")))
    return float("nan")
def select_exhaustive_top_configs(
    search_result: dict,
    *,
    top_n: int = 150,
    r_hat_max: float = 1.02,
    p_tau_gt_min: float = 0.5,
    tau_gt_threshold: float = 6.0,
    sort_by: str = "loo",
) -> tuple[list[dict], pd.DataFrame]:
    """Select top-N exhaustive-search configs by elpd_loo with r_hat and P(tau>threshold) gates."""
    results = [r for r in (search_result.get("results") or []) if r.get("status") == "ok"]
    filtered: list[dict] = []
    for rec in results:
        rh = float(rec.get("r_hat_max", float("inf")))
        p_gt = _exhaustive_p_tau_gt(rec, tau_gt_threshold)
        if rh <= float(r_hat_max) and math.isfinite(p_gt) and p_gt >= float(p_tau_gt_min):
            filtered.append(rec)

    key = str(sort_by).strip().lower()
    if key not in {"loo", "waic", "tau_map", "tau_map_concentration"}:
        raise ValueError("sort_by must be one of: loo, waic, tau_map, tau_map_concentration")

    def _sort_val(rec: dict) -> float:
        v = float(rec.get(key, float("nan")))
        return v if math.isfinite(v) else float("-inf")

    filtered = sorted(filtered, key=_sort_val, reverse=True)
    if top_n is not None:
        filtered = filtered[: max(1, int(top_n))]

    configs = [_clone_config(rec["config"]) for rec in filtered if rec.get("config")]
    rows: list[dict[str, Any]] = []
    for rank, rec in enumerate(filtered, start=1):
        cfg = rec.get("config") or {}
        rem = cfg.get("rem_profile_params") or {}
        rows.append(
            {
                "rank": rank,
                "fingerprint": str(rec.get("fingerprint", "")),
                "features": _config_feature_label(cfg),
                "n_points": int(rem.get("n_points_per_day", rem.get("window_size_hours", -1))),
                "overlap": float(rem.get("overlap", -1.0)),
                "step_h": int(rem.get("step_size_hours", -1)),
                "elpd_loo": float(rec.get("elpd_loo", rec.get("loo", float("nan")))),
                "waic": float(rec.get("waic", float("nan"))),
                "tau_map": int(rec.get("tau_map", -1)),
                "P_tau_map": float(rec.get("tau_map_concentration", float("nan"))),
                "p_tau_gt_6": _exhaustive_p_tau_gt(rec, tau_gt_threshold),
                "r_hat_max": float(rec.get("r_hat_max", float("nan"))),
            }
        )
    return configs, pd.DataFrame(rows)
def summarize_exhaustive_search(search_result: dict) -> dict:
    """Compute summary statistics from exhaustive search results.

    ``best_loo`` and nested ``loo`` fields use ``elpd_loo`` scale (same as ArviZ
    ``elpd_loo``), not LOO-IC, for exhaustive search runs.
    """
    from collections import Counter

    results = list(search_result.get("results") or [])
    valid = [
        r
        for r in results
        if r.get("status") == "ok"
        and math.isfinite(float(r.get("loo", float("nan"))))
        and float(r.get("r_hat_max", float("inf"))) <= 1.05
        and float(r.get("ess_min_bulk", float("-inf"))) >= 100.0
        and int(r.get("n_divergences", 1)) == 0
    ]
    valid_sorted = sorted(valid, key=lambda x: float(x.get("loo", float("-inf"))), reverse=True)

    best = valid_sorted[0] if valid_sorted else None
    top10 = valid_sorted[:10]
    top20 = valid_sorted[:20]

    feature_counter: Counter[str] = Counter()
    like_counter: dict[str, Counter[str]] = {}
    for rec in valid:
        cfg = rec.get("config") or {}
        fs = cfg.get("feature_selection") or {}
        for _, metrics in fs.items():
            for metric_name in metrics:
                mk = str(metric_name)
                feature_counter[mk] += 1
        ps = cfg.get("parameter_selection") or {}
        for metric_name, spec in ps.items():
            mk = str(metric_name)
            lk = str((spec or {}).get("likelihood", ""))
            like_counter.setdefault(mk, Counter())
            like_counter[mk][lk] += 1

    n_local_optima = 0
    if top20:
        configs = [rec.get("config") or {} for rec in top20]
        dist = compute_model_distance_matrix(configs)
        n = dist.shape[0]
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            n_local_optima += 1
            stack = [i]
            visited[i] = True
            while stack:
                cur = stack.pop()
                neigh = np.where(dist[cur] <= 2)[0]
                for nb in neigh:
                    j = int(nb)
                    if not visited[j]:
                        visited[j] = True
                        stack.append(j)

    loo_vals_top20 = [float(r.get("loo", float("nan"))) for r in top20]
    loo_vals_top20 = [v for v in loo_vals_top20 if math.isfinite(v)]
    if loo_vals_top20:
        loo_range_top20 = (float(min(loo_vals_top20)), float(max(loo_vals_top20)))
    else:
        loo_range_top20 = (float("nan"), float("nan"))

    tau_mode = None
    if top10:
        tau_counts = Counter(int(r.get("tau_map", -1)) for r in top10 if int(r.get("tau_map", -1)) >= 0)
        if tau_counts:
            tau_mode = int(tau_counts.most_common(1)[0][0])

    best_loo = float(best.get("loo", float("nan"))) if best else float("nan")
    best_waic = float(best.get("waic", float("nan"))) if best else float("nan")
    best_fp = str(best.get("fingerprint")) if best else None

    return {
        "n_total": int(search_result.get("n_total", 0)),
        "n_fitted": int(search_result.get("n_fitted", 0)),
        "n_filtered": int(search_result.get("n_filtered", 0)),
        "n_valid": int(len(valid)),
        "best_loo": best_loo,
        "best_elpd_loo": best_loo,
        "best_waic": best_waic,
        "best_config_fingerprint": best_fp,
        "top_fingerprints_by_loo": [
            {
                "fingerprint": str(r.get("fingerprint")),
                "elpd_loo": float(r.get("elpd_loo", r.get("loo", float("nan")))),
            }
            for r in valid_sorted[:10]
        ],
        "feature_visit_freq": dict(feature_counter),
        "likelihood_visit_freq": {
            metric_name: dict(counter)
            for metric_name, counter in like_counter.items()
        },
        "n_local_optima": int(n_local_optima),
        "loo_range_top20": loo_range_top20,
        "tau_map_mode": tau_mode,
        "top_tau_by_loo": [
            {
                "fingerprint": str(r.get("fingerprint")),
                "elpd_loo": float(r.get("elpd_loo", r.get("loo", float("nan")))),
                "tau_map": int(r.get("tau_map", -1)),
                "tau_map_concentration": float(r.get("tau_map_concentration", float("nan"))),
                "features": _config_feature_label(r.get("config") or {}),
            }
            for r in top10
        ],
    }
from seismic_pipeline.bayesian.search_export import export_exhaustive_search_results_to_csv
