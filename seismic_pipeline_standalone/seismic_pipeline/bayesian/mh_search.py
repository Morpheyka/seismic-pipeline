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
    _RUNTIME_LAST_EXPORT_CFG,
    _RUNTIME_LAST_PREPARE_CFG,
)
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    prepare_model_data,
    set_runtime_data_norm,
)
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only
from seismic_pipeline.bayesian.diagnostics import score_changepoint_trace, changepoint_log_target
from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.search import propose_changepoint_model_config
from seismic_pipeline.bayesian.search_common import _fit_config_once
from seismic_pipeline.visualization.changepoint_ppc import plot_posterior_predictive_check

def _running_log_target_quantiles(
    chain_records: List[dict[str, Any]],
    window: int,
) -> tuple[float, float, float]:
    w = max(1, int(window))
    vals: List[float] = []
    for r in chain_records[-w:]:
        lt = r.get("log_target")
        if lt is None:
            continue
        v = float(lt)
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(np.quantile(a, 0.1)), float(np.quantile(a, 0.5)), float(np.quantile(a, 0.9))
def _best_fingerprint_by_elpd(score_cache: dict[str, dict[str, Any]]) -> tuple[str | None, float]:
    best_fp: str | None = None
    best_elpd = float("-inf")
    for fp, sc in score_cache.items():
        e = float(sc.get("elpd", float("-inf")))
        if math.isfinite(e) and e > best_elpd:
            best_elpd = e
            best_fp = str(fp)
    if best_fp is None:
        return None, float("nan")
    return best_fp, best_elpd
def _model_log_prior_from_score(
    score_parts: dict[str, Any],
    *,
    model_prior_type: str = "bic",
    model_prior_lambda: float = 0.69,
) -> float:
    prior_type = str(model_prior_type).strip().lower()
    p = max(int(score_parts.get("n_feature_blocks", 0)), 1)
    if prior_type == "uniform":
        return 0.0
    if prior_type == "inverse":
        return -math.log(float(p))
    if prior_type == "bic":
        n_observations = max(float(score_parts.get("n_observations", 1.0)), 1.0)
        return -0.5 * float(p) * math.log(n_observations)
    if prior_type == "lambda":
        return -float(model_prior_lambda) * float(p)
    raise ValueError(
        f"Unknown model_prior_type={model_prior_type!r}; "
        "use one of: 'uniform', 'inverse', 'bic', 'lambda'."
    )
def check_mh_convergence(
    history: List[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[bool, str | None]:
    if len(history) <= 1:
        return False, None

    patience = max(1, int(config.get("patience", 30)))
    window = max(2, int(config.get("window", 50)))
    tol_mean = float(config.get("tol_mean", 0.1))
    saturation_threshold = float(config.get("saturation_threshold", 0.1))

    best_log_target = float("-inf")
    best_iteration = 0
    for rec in history:
        lt = rec.get("log_target")
        if lt is None:
            continue
        v = float(lt)
        if math.isfinite(v) and v > best_log_target + 1e-12:
            best_log_target = v
            best_iteration = int(rec.get("iteration", 0))
    cur_iteration = int(history[-1].get("iteration", 0))
    if cur_iteration - best_iteration >= patience:
        return True, "no_log_target_improvement"

    recent = history[-window:]
    recent_vals: List[float] = []
    for rec in recent:
        lt = rec.get("log_target")
        if lt is None:
            continue
        v = float(lt)
        if math.isfinite(v):
            recent_vals.append(v)
    if len(recent_vals) >= 4:
        split = len(recent_vals) // 2
        if split > 0:
            mean_old = float(np.mean(np.asarray(recent_vals[:split], dtype=float)))
            mean_new = float(np.mean(np.asarray(recent_vals[split:], dtype=float)))
            if abs(mean_new - mean_old) <= tol_mean:
                return True, "log_target_stabilized"

    seen_fingerprints: set[str] = set()
    new_model_flags: List[float] = []
    for rec in history:
        fp = rec.get("fingerprint")
        if not fp:
            new_model_flags.append(0.0)
            continue
        key = str(fp)
        if key in seen_fingerprints:
            new_model_flags.append(0.0)
        else:
            seen_fingerprints.add(key)
            new_model_flags.append(1.0)
    if new_model_flags:
        recent_new = np.asarray(new_model_flags[-window:], dtype=float)
        if recent_new.size > 0 and float(np.mean(recent_new)) < saturation_threshold:
            return True, "model_space_saturated"

    return False, None
def metropolis_hastings_model_search(
    *,
    initial_config: dict,
    proposal_options: dict,
    data_norm: np.ndarray,
    n_iterations: int | None = None,
    min_iterations: int = 100,
    max_iterations: int = 500,
    patience: int = 30,
    window: int = 50,
    tol_mean: float = 0.1,
    saturation_threshold: float = 0.1,
    model_prior_type: str = "bic",
    model_prior_lambda: float = 0.69,
    draws: int = 800,
    tune: int = 1200,
    nuts_backend: str = "pymc",
    chains: int = 2,
    cores: int | None = None,
    tau_mode: str = "marginalized",
    tau_lower: int = 2,
    tau_upper: int | None = None,
    rem_profile_params: dict | None = None,
    cache_fits: bool = True,
    seed: int | None = None,
    target_weights: dict[str, float] | None = None,
    verbose: bool = False,
    verbose_every: int = 10,
    show_progress_bar: bool = True,
    ic_criterion: str = "waic",
    quantile_window: int = 20,
    n_mh_chains: int = 1,
    run_ppc_for_best: bool = False,
    ppc_observed_data: np.ndarray | dict[str, np.ndarray] | None = None,
    ppc_num_pp_samples: int = 300,
    ppc_random_seed: int | None = None,
    precompute_features: bool = False,
) -> dict[str, Any]:
    """Outer Metropolis-Hastings over discrete model configs; inner NUTS per accepted evaluation.

    Each iteration proposes one neighboring model, fits it with PyMC, and accepts/rejects
    using ``changepoint_log_target`` on ``score_changepoint_trace`` outputs (WAIC/LOO elpd
    when a PyMC ``model`` is available from the fit step).

    ``verbose_every`` (default 10): when ``verbose`` is True, print MH progress every N iterations.

    When ``precompute_features=True``, REM profiles and chunk features for all combinations
    in ``proposal_options`` are computed once up front; the MH loop uses cached lookups only.
    """
    tw = target_weights or {}
    runtime_export_cfg_base = dict(_RUNTIME_LAST_EXPORT_CFG) if _RUNTIME_LAST_EXPORT_CFG is not None else None
    base_export_output_dir = (
        str(runtime_export_cfg_base.get("output_dir", "."))
        if runtime_export_cfg_base is not None
        else None
    )
    precomputed_features: dict[tuple, np.ndarray] | None = None
    precompute_seconds = 0.0
    if precompute_features:
        t_pre0 = time.perf_counter()
        precompute_cfg = {
            "proposal_options": proposal_options,
            "rem_profile_params": rem_profile_params or initial_config.get("rem_profile_params"),
        }
        precomputed_features = precompute_all_features(data_norm, precompute_cfg)
        precompute_seconds = time.perf_counter() - t_pre0
        if verbose:
            print(
                f"[MH] precompute_all_features: {precompute_seconds:.2f}s "
                f"({len(precomputed_features)} feature arrays)",
                flush=True,
            )
    n_mh_chains = max(1, int(n_mh_chains))
    ve = max(1, int(verbose_every))
    early_stopping_enabled = n_iterations is None
    if n_iterations is not None:
        fixed_steps = max(0, int(n_iterations))
        min_iterations_cfg = fixed_steps
        max_iterations_cfg = fixed_steps
    else:
        min_iterations_cfg = max(0, int(min_iterations))
        max_iterations_cfg = max(1, int(max_iterations))
        if min_iterations_cfg > max_iterations_cfg:
            min_iterations_cfg = max_iterations_cfg
    stop_config = {
        "patience": max(1, int(patience)),
        "window": max(2, int(window)),
        "tol_mean": float(tol_mean),
        "saturation_threshold": float(saturation_threshold),
    }

    def _final_log_target(chain_result: dict[str, Any]) -> float:
        ch = list(chain_result.get("chain") or [])
        if not ch:
            return float("-inf")
        return float(ch[-1].get("log_target", float("-inf")))

    def run_one_chain(seed_local: int | None, chain_idx: int) -> dict[str, Any]:
        t_sampling0 = time.perf_counter()
        rng = np.random.default_rng(seed_local)
        current = _clone_config(initial_config)
        if rem_profile_params is not None:
            current["rem_profile_params"] = dict(rem_profile_params)
        pbar = None
        if show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(
                    total=max_iterations_cfg + 1,
                    desc=f"MH chain {chain_idx + 1}/{n_mh_chains}",
                    leave=True,
                )
            except Exception:
                pbar = None

        def _fmt_ic(v: Any) -> str:
            try:
                fv = float(v)
            except Exception:
                return "nan"
            return f"{fv:.3f}" if math.isfinite(fv) else "nan"

        def _print_ic_summary(it: int, fp: str, sc: dict[str, Any]) -> None:
            if not verbose:
                return
            print(
                f"[MH chain {chain_idx}] iter={it} fit summary fingerprint={fp} "
                f"waic={_fmt_ic(sc.get('waic'))} loo={_fmt_ic(sc.get('loo'))} "
                f"waic_warning={bool(sc.get('waic_warning_flag', False))}",
                flush=True,
            )

        score_cache: dict[str, dict[str, Any]] = {}
        trace_cache: dict[str, Any] = {}
        model_cache: dict[str, Any] = {}
        group_data_cache: dict[str, dict] = {}
        config_cache: dict[str, dict[str, Any]] = {}
        rem_data_cache: dict[str, np.ndarray] = {}

        def ensure_data_for_config(cfg: dict) -> np.ndarray:
            """Return data_norm aligned with cfg['rem_profile_params'], with per-REM caching."""
            nonlocal data_norm
            if precompute_features:
                return data_norm
            rpp = cfg.get("rem_profile_params")
            if not rpp or runtime_export_cfg_base is None:
                return data_norm
            rem_key = json.dumps(
                {
                    "window_size_hours": int(rpp["window_size_hours"]),
                    "step_size_hours": int(rpp["step_size_hours"]),
                    "rem_stage": int(rpp["rem_stage"]),
                },
                sort_keys=True,
            )
            if rem_key in rem_data_cache:
                return rem_data_cache[rem_key]
            export_cfg = dict(runtime_export_cfg_base)
            if base_export_output_dir is not None:
                export_cfg["output_dir"] = os.path.join(
                    base_export_output_dir,
                    (
                        f"rem_w{int(rpp['window_size_hours'])}"
                        f"_s{int(rpp['step_size_hours'])}"
                        f"_stage{int(rpp['rem_stage'])}"
                    ),
                )
            export_cfg.update(
                {
                    "window_size_hours": int(rpp["window_size_hours"]),
                    "step_size_hours": int(rpp["step_size_hours"]),
                    "rem_stage": int(rpp["rem_stage"]),
                }
            )
            export_result = export_rem_profiles_10days_cached_only(**export_cfg)
            prep_cfg = _RUNTIME_LAST_PREPARE_CFG or {}
            prep_csv_path = export_result["paths"]["nanpad_output_csv"]
            prep = prepare_model_data(
                csv_path=prep_csv_path,
                bad_sample_indices=prep_cfg.get("bad_sample_indices"),
            )
            set_runtime_data_norm(prep["data_norm"])
            rem_data_cache[rem_key] = prep["data_norm"]
            return prep["data_norm"]

        chain_records: List[dict[str, Any]] = []
        fp0 = changepoint_model_config_fingerprint(current)
        data_work = ensure_data_for_config(current)
        gd0, tr0, sp0, m0 = _fit_config_once(
            current,
            data_norm=data_work,
            draws=draws,
            tune=tune,
            nuts_backend=nuts_backend,
            chains=chains,
            cores=cores,
            tau_mode=tau_mode,
            tau_lower=tau_lower,
            tau_upper=tau_upper,
            ic_criterion=ic_criterion,
            sampler_progressbar=False,
            precomputed_features=precomputed_features,
        )
        _print_ic_summary(0, fp0, sp0)
        score_cache[fp0] = sp0
        config_cache[fp0] = _clone_config(current)
        if cache_fits:
            trace_cache[fp0] = tr0
            model_cache[fp0] = m0
            group_data_cache[fp0] = gd0
        current_score = sp0
        log_cur = changepoint_log_target(sp0, **tw)
        log_prior_cur = _model_log_prior_from_score(
            sp0,
            model_prior_type=model_prior_type,
            model_prior_lambda=model_prior_lambda,
        )

        rec0: dict[str, Any] = {
            "iteration": 0,
            "fingerprint": fp0,
            "config": _clone_config(current),
            "accepted": True,
            "log_target": log_cur,
            "log_prior": log_prior_cur,
            "log_posterior_target": log_cur + log_prior_cur,
            "score": sp0,
        }
        chain_records.append(rec0)
        q10, q50, q90 = _running_log_target_quantiles(chain_records, quantile_window)
        chain_records[-1].update(
            {"log_target_q10": q10, "log_target_q50": q50, "log_target_q90": q90}
        )
        if pbar is not None:
            pbar.update(1)
        if verbose:
            print(
                f"[MH chain {chain_idx}] iter=0 initial fit done "
                f"log_target={log_cur:.3f} fingerprint={fp0}",
                flush=True,
            )

        n_accept = 0
        stopping_reason: str | None = None
        for it in range(1, max_iterations_cfg + 1):
            proposed = propose_changepoint_model_config(current, proposal_options, rng)
            proposed.setdefault("tau_threshold", current.get("tau_threshold", 7.0))
            fp = changepoint_model_config_fingerprint(proposed)
            if cache_fits and fp in score_cache:
                sp_star = score_cache[fp]
                log_star = changepoint_log_target(sp_star, **tw)
                log_prior_star = _model_log_prior_from_score(
                    sp_star,
                    model_prior_type=model_prior_type,
                    model_prior_lambda=model_prior_lambda,
                )
            else:
                try:
                    dnorm = ensure_data_for_config(proposed)
                    gd_star, tr_star, sp_star, m_star = _fit_config_once(
                        proposed,
                        data_norm=dnorm,
                        draws=draws,
                        tune=tune,
                        nuts_backend=nuts_backend,
                        chains=chains,
                        cores=cores,
                        tau_mode=tau_mode,
                        tau_lower=tau_lower,
                        tau_upper=tau_upper,
                        ic_criterion=ic_criterion,
                        sampler_progressbar=False,
                        precomputed_features=precomputed_features,
                    )
                    _print_ic_summary(it, fp, sp_star)
                    score_cache[fp] = sp_star
                    config_cache[fp] = _clone_config(proposed)
                    if cache_fits:
                        trace_cache[fp] = tr_star
                        model_cache[fp] = m_star
                        group_data_cache[fp] = gd_star
                except Exception as exc:
                    if verbose:
                        print(f"[MH iter {it}] proposal rejected (build/sample error): {exc}")
                    chain_records.append(
                        {
                            "iteration": it,
                            "fingerprint": fp,
                            "config": _clone_config(proposed),
                            "accepted": False,
                            "log_target": float("-inf"),
                            "log_prior": float("nan"),
                            "log_posterior_target": float("-inf"),
                            "score": None,
                            "error": str(exc),
                        }
                    )
                    q10, q50, q90 = _running_log_target_quantiles(chain_records, quantile_window)
                    chain_records[-1].update(
                        {"log_target_q10": q10, "log_target_q50": q50, "log_target_q90": q90}
                    )
                    if pbar is not None:
                        pbar.update(1)
                    if early_stopping_enabled and it >= min_iterations_cfg:
                        should_stop, reason = check_mh_convergence(chain_records, stop_config)
                        if should_stop:
                            stopping_reason = reason
                            if verbose:
                                print(
                                    f"[MH chain {chain_idx}] stopping at iter={it}: {reason}",
                                    flush=True,
                                )
                            break
                    continue
                log_star = changepoint_log_target(sp_star, **tw)
                log_prior_star = _model_log_prior_from_score(
                    sp_star,
                    model_prior_type=model_prior_type,
                    model_prior_lambda=model_prior_lambda,
                )

            log_accept = (log_star - log_cur) + (log_prior_star - log_prior_cur)
            if math.log(rng.random()) < log_accept:
                current = _clone_config(proposed)
                log_cur = log_star
                log_prior_cur = log_prior_star
                current_score = sp_star
                n_accept += 1
                acc = True
            else:
                acc = False

            cur_fp = changepoint_model_config_fingerprint(current)
            chain_records.append(
                {
                    "iteration": it,
                    "fingerprint": cur_fp,
                    "config": _clone_config(current),
                    "accepted": acc,
                    "log_target": log_cur,
                    "log_prior": log_prior_cur,
                    "log_posterior_target": log_cur + log_prior_cur,
                    "score": current_score,
                }
            )
            q10, q50, q90 = _running_log_target_quantiles(chain_records, quantile_window)
            chain_records[-1].update(
                {"log_target_q10": q10, "log_target_q50": q50, "log_target_q90": q90}
            )
            if pbar is not None:
                pbar.update(1)
            if verbose and it % ve == 0:
                print(
                    f"[MH chain {chain_idx}] iter={it} accept_rate~{n_accept / it:.3f} "
                    f"log_target={log_cur:.3f} log_target_q10/50/90={q10:.3f}/{q50:.3f}/{q90:.3f}",
                    flush=True,
                )
            if early_stopping_enabled and it >= min_iterations_cfg:
                should_stop, reason = check_mh_convergence(chain_records, stop_config)
                if should_stop:
                    stopping_reason = reason
                    if verbose:
                        print(
                            f"[MH chain {chain_idx}] stopping at iter={it}: {reason}",
                            flush=True,
                        )
                    break

        if stopping_reason is None:
            stopping_reason = "max_iterations_reached"
        n_iterations_run = max(0, len(chain_records) - 1)
        if pbar is not None:
            pbar.close()

        sampling_seconds = time.perf_counter() - t_sampling0
        return {
            "chain": chain_records,
            "acceptance_rate": n_accept / max(n_iterations_run, 1),
            "score_cache": score_cache,
            "trace_cache": trace_cache if cache_fits else {},
            "model_cache": model_cache if cache_fits else {},
            "group_data_cache": group_data_cache if cache_fits else {},
            "config_cache": config_cache,
            "final_config": _clone_config(current),
            "stopping_reason": stopping_reason,
            "n_iterations_run": n_iterations_run,
            "sampling_seconds": sampling_seconds,
        }

    chain_results: List[dict[str, Any]] = []
    for ci in range(n_mh_chains):
        seed_c = None if seed is None else int(seed) + ci
        if verbose and n_mh_chains > 1:
            print(f"[MH] starting outer chain {ci + 1}/{n_mh_chains}", flush=True)
        chain_results.append(run_one_chain(seed_c, ci))

    best = max(chain_results, key=_final_log_target)
    out = dict(best)
    best_fp, best_elpd = _best_fingerprint_by_elpd(out["score_cache"])
    out["best_fingerprint"] = best_fp
    out["best_elpd"] = best_elpd

    total_sampling_seconds = float(
        sum(float(cr.get("sampling_seconds", 0.0)) for cr in chain_results)
    )
    out["precompute_seconds"] = precompute_seconds
    out["sampling_seconds"] = total_sampling_seconds
    if precomputed_features is not None:
        out["precomputed_features"] = precomputed_features
    if verbose and precompute_features:
        total_s = precompute_seconds + total_sampling_seconds
        pct_pre = 100.0 * precompute_seconds / total_s if total_s > 0 else 0.0
        pct_samp = 100.0 * total_sampling_seconds / total_s if total_s > 0 else 0.0
        print(
            f"[MH] timing: precompute={precompute_seconds:.2f}s ({pct_pre:.1f}%), "
            f"sampling={total_sampling_seconds:.2f}s ({pct_samp:.1f}%)",
            flush=True,
        )

    if n_mh_chains > 1:
        finals = [_final_log_target(cr) for cr in chain_results]
        out["mh_chain_results"] = chain_results
        out["mh_chain_stopping_reasons"] = [str(cr.get("stopping_reason", "")) for cr in chain_results]
        out["final_log_targets"] = finals
        fa = np.asarray([x for x in finals if math.isfinite(x)], dtype=float)
        out["final_log_target_std"] = float(np.std(fa)) if fa.size > 1 else 0.0

    if run_ppc_for_best:
        best_trace = None
        best_model = None
        if best_fp is not None:
            best_trace = (out.get("trace_cache") or {}).get(best_fp)
            best_model = (out.get("model_cache") or {}).get(best_fp)
        if best_trace is not None and best_model is not None:
            best_gd = (out.get("group_data_cache") or {}).get(best_fp)
            best_cfg = (out.get("config_cache") or {}).get(best_fp)
            out["best_model_ppc"] = plot_posterior_predictive_check(
                best_trace,
                best_model,
                observed_data=ppc_observed_data,
                group_data=best_gd,
                parameter_selection=best_cfg.get("parameter_selection") if best_cfg else None,
                num_pp_samples=ppc_num_pp_samples,
                random_seed=ppc_random_seed,
            )
        else:
            warnings.warn(
                "metropolis_hastings_model_search: run_ppc_for_best=True but best trace/model "
                "is unavailable (likely cache_fits=False or no best fingerprint).",
                UserWarning,
                stacklevel=2,
            )
    return out
def summarize_model_search(search_result: dict) -> dict[str, Any]:
    """Aggregate MH chain: model visit counts and visit frequencies for metrics/groups/n_chunks."""
    chain = list(search_result.get("chain") or [])
    if not chain:
        return {
            "model_visit_counts": {},
            "top_fingerprints_by_log_target": [],
            "feature_visit_freq": {},
            "group_visit_freq": {},
            "n_chunks_visit_freq": {},
            "likelihood_visit_freq": {},
            "top_fingerprints_by_elpd": [],
            "elpd_by_fingerprint": {},
            "stopping_reason": search_result.get("stopping_reason"),
            "n_iterations_run": int(search_result.get("n_iterations_run", 0)),
            "final_log_target": float("nan"),
            "unique_visited_models": 0,
            "accepted_proposals": 0,
            "acceptance_rate": float(search_result.get("acceptance_rate", 0.0)),
            "final_log_target_std": float("nan"),
            "final_log_targets": [],
        }

    from collections import Counter

    visit_fp: List[str] = []
    w_feat: dict[str, float] = {}
    w_group: dict[str, float] = {}
    w_n_chunks: dict[str, float] = {}
    w_like: dict[str, dict[str, float]] = {}
    log_targets: dict[str, float] = {}
    elpd_by_fp: dict[str, float] = {}

    for rec in chain:
        fp = rec.get("fingerprint")
        if not fp:
            continue
        fp = str(fp)
        visit_fp.append(fp)
        sc = rec.get("score") or {}
        lt = float(rec.get("log_target", float("-inf")))
        if math.isfinite(lt):
            log_targets[fp] = max(log_targets.get(fp, float("-inf")), lt)
        elp = sc.get("elpd")
        if elp is not None and math.isfinite(float(elp)):
            efv = float(elp)
            elpd_by_fp[fp] = max(elpd_by_fp.get(fp, float("-inf")), efv)
        for feat in sc.get("active_features") or []:
            w_feat[feat] = w_feat.get(feat, 0.0) + 1.0
        cfg = rec.get("config") or {}
        fs_cfg = cfg.get("feature_selection")
        if isinstance(fs_cfg, dict):
            for group_name, feats in fs_cfg.items():
                if not feats:
                    continue
                gk = str(group_name)
                w_group[gk] = w_group.get(gk, 0.0) + 1.0
        n_chunks_val = cfg.get("n_chunks")
        if n_chunks_val is not None:
            try:
                nk = int(n_chunks_val)
                n_key = str(nk)
                w_n_chunks[n_key] = w_n_chunks.get(n_key, 0.0) + 1.0
            except Exception:
                pass
        likes = sc.get("likelihoods") or {}
        for feat, lk in likes.items():
            w_like.setdefault(feat, {})
            w_like[feat][lk] = w_like[feat].get(lk, 0.0) + 1.0

    score_cache = search_result.get("score_cache") or {}
    for fp, sc in score_cache.items():
        if not isinstance(sc, dict):
            continue
        fp = str(fp)
        elp = sc.get("elpd")
        if elp is None or not math.isfinite(float(elp)):
            continue
        efv = float(elp)
        elpd_by_fp[fp] = max(elpd_by_fp.get(fp, float("-inf")), efv)

    counts = Counter(visit_fp)
    top = sorted(log_targets.keys(), key=lambda f: log_targets[f], reverse=True)[:15]
    top_elpd = sorted(elpd_by_fp.keys(), key=lambda f: elpd_by_fp[f], reverse=True)[:15]

    def _renorm(d: dict[str, float]) -> dict[str, float]:
        s = sum(d.values()) or 1.0
        return {k: v / s for k, v in sorted(d.items(), key=lambda kv: -kv[1])}

    def _renorm_n_chunks(d: dict[str, float]) -> dict[str, float]:
        s = sum(d.values()) or 1.0
        return {
            k: v / s
            for k, v in sorted(
                d.items(),
                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9,
            )
        }

    like_renorm = {feat: _renorm(vs) for feat, vs in w_like.items()}
    accepted_proposals = int(sum(1 for r in chain[1:] if bool(r.get("accepted", False))))
    n_steps = max(0, len(chain) - 1)
    acceptance_rate = float(search_result.get("acceptance_rate", accepted_proposals / max(n_steps, 1)))
    out: dict[str, Any] = {
        "model_visit_counts": dict(counts.most_common(25)),
        "top_fingerprints_by_log_target": top,
        "log_target_by_fingerprint": log_targets,
        "feature_visit_freq": _renorm(w_feat),
        "group_visit_freq": _renorm(w_group),
        "n_chunks_visit_freq": _renorm_n_chunks(w_n_chunks),
        "likelihood_visit_freq": like_renorm,
        "mean_acceptance_indicator": float(np.mean([1.0 if r.get("accepted") else 0.0 for r in chain[1:]]))
        if len(chain) > 1
        else 0.0,
        "top_fingerprints_by_elpd": top_elpd,
        "elpd_by_fingerprint": elpd_by_fp,
        "stopping_reason": search_result.get("stopping_reason"),
        "n_iterations_run": int(search_result.get("n_iterations_run", n_steps)),
        "final_log_target": float(chain[-1].get("log_target", float("nan"))),
        "unique_visited_models": len(set(visit_fp)),
        "accepted_proposals": accepted_proposals,
        "acceptance_rate": acceptance_rate,
        "final_log_target_std": float(search_result.get("final_log_target_std", float("nan"))),
        "final_log_targets": list(search_result.get("final_log_targets") or []),
    }
    return out
