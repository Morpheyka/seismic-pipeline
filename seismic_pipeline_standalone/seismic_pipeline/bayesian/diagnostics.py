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
from xarray import DataTree
import xarray as xr

from seismic_pipeline.bayesian.priors import _parse_parameter_selection

def _is_inferencedata(trace) -> bool:
    """True for ArviZ DataTree traces (PyMC default since ArviZ 1.x)."""
    return isinstance(trace, DataTree)
def _available_varnames(trace) -> set[str]:
    if _is_inferencedata(trace):
        return set(trace.posterior.data_vars)
    return set(trace.varnames)
def _values_flat(trace, var_name: str) -> np.ndarray:
    if _is_inferencedata(trace):
        return np.concatenate(
            [c.reshape(-1) for c in _values_by_chain(trace, var_name)],
            axis=0,
        )
    return np.asarray(trace[var_name]).reshape(-1)
def _values_by_chain(trace, var_name: str) -> list[np.ndarray]:
    if _is_inferencedata(trace):
        da = trace.posterior[var_name]
        return [
            np.asarray(da.isel(chain=i).values, dtype=np.float64)
            for i in range(int(da.sizes["chain"]))
        ]
    return trace.get_values(var_name, combine=False)


def _loglik_numpy_from_posterior(trace: DataTree, var_name: str) -> np.ndarray:
    """Materialize pointwise log-likelihood without syncing the full JAX posterior."""
    da = trace.posterior[var_name]
    ll = np.stack(
        [
            np.asarray(da.isel(chain=i).values, dtype=np.float64)
            for i in range(int(da.sizes["chain"]))
        ],
        axis=0,
    )
    if ll.ndim == 2:
        ll = ll[..., np.newaxis]
    return ll


def _attach_loglik_idata_from_posterior(trace: DataTree, var_name: str) -> DataTree:
    """Attach log_likelihood group without copying the full posterior to numpy."""
    ll = _loglik_numpy_from_posterior(trace, var_name)
    dims = list(trace.posterior[var_name].dims)
    idata = trace.copy(deep=False)
    idata["log_likelihood"] = xr.Dataset({var_name: (dims, ll)})
    return idata


def _warm_jax_xarray_dataset(
    ds: xr.Dataset,
    preferred_vars: Iterable[str] | None = None,
) -> None:
    """Trigger a single JAX→host sync before bulk numpy conversion."""
    if preferred_vars:
        for name in preferred_vars:
            if name in ds:
                da = ds[name]
                if "chain" in da.dims:
                    np.asarray(da.isel(chain=0).values)
                else:
                    np.asarray(da.values)
                return
    for name in ("tau_mean", "changepoint_joint_log_lik", "lp"):
        if name in ds:
            da = ds[name]
            if "chain" in da.dims:
                np.asarray(da.isel(chain=0).values)
            else:
                np.asarray(da.values)
            return
    if ds.data_vars:
        da = next(iter(ds.data_vars.values()))
        if "chain" in da.dims:
            np.asarray(da.isel(chain=0).values)
        else:
            np.asarray(da.values)


def _xarray_dataset_to_numpy(
    ds: xr.Dataset,
    *,
    var_names: Iterable[str] | None = None,
) -> xr.Dataset:
    allowed = set(var_names) if var_names is not None else None
    converted: dict[str, xr.DataArray] = {}
    for name in ds.data_vars:
        if allowed is not None and name not in allowed:
            continue
        da = ds[name]
        converted[name] = xr.DataArray(
            np.asarray(da.values, dtype=np.float64),
            dims=da.dims,
            coords=da.coords,
        )
    return xr.Dataset(converted)


def materialize_inferencedata_numpy(
    trace: DataTree,
    *,
    posterior_var_names: Iterable[str] | None = None,
    include_sample_stats: bool = True,
) -> DataTree:
    """Materialize JAX-backed posterior/sample_stats to host numpy arrays.

    JAX backends return lazy arrays; the first ``.values`` access evaluates the
  full trace graph (~10–20s). Doing that once here avoids repeated sync during
    LOO scoring and diagnostics.
    """
    if not _is_inferencedata(trace):
        return trace
    out = trace.copy(deep=False)
    if trace.posterior.data_vars:
        selected_names = None if posterior_var_names is None else set(posterior_var_names)
        warm_names = (
            selected_names
            if selected_names is not None
            else ("tau_mean", "changepoint_joint_log_lik")
        )
        _warm_jax_xarray_dataset(trace.posterior, preferred_vars=warm_names)
        out.posterior = _xarray_dataset_to_numpy(
            trace.posterior,
            var_names=selected_names,
        )
    if include_sample_stats and hasattr(trace, "sample_stats") and trace.sample_stats.data_vars:
        out.sample_stats = _xarray_dataset_to_numpy(trace.sample_stats)
    return out


def _sampler_stat(trace, stat_name: str):
    if _is_inferencedata(trace):
        if hasattr(trace, "sample_stats") and stat_name in trace.sample_stats:
            return np.asarray(trace.sample_stats[stat_name])
        raise KeyError(f"Sampler stat '{stat_name}' not found in trace.sample_stats")
    return trace.get_sampler_stats(stat_name, combine=True)
def summary_from_trace(trace, var_names):
    """Build ArviZ summary (mean, sd, r_hat, ESS)."""
    if _is_inferencedata(trace):
        try:
            return az.summary(trace, var_names=var_names)
        except (AttributeError, TypeError, ValueError):
            return az.summary(trace.posterior, var_names=var_names)

    posterior = {}
    for var in var_names:
        chains = _values_by_chain(trace, var)
        posterior[var] = np.stack(chains, axis=0)
    idata = az.from_dict({"posterior": posterior})
    return az.summary(idata, var_names=var_names)
def tau_probabilities(trace):
    """Return tau support and probabilities P(tau=k)."""
    trace_vars = _available_varnames(trace)
    if "tau" in trace_vars:
        tau_values = _values_flat(trace, "tau").astype(int).ravel()
        tau_min, tau_max = int(tau_values.min()), int(tau_values.max())
        support = np.arange(tau_min, tau_max + 1)
        counts = np.bincount(tau_values, minlength=tau_max + 1)[tau_min : tau_max + 1]
        probs = counts / counts.sum()
        return support, probs

    if "tau_probs" in trace_vars and "tau_support" in trace_vars:
        if _is_inferencedata(trace):
            probs_draws = np.stack(_values_by_chain(trace, "tau_probs"), axis=0)
            support_draws = np.stack(_values_by_chain(trace, "tau_support"), axis=0)
            probs = probs_draws.mean(axis=(0, 1))
            support = support_draws[0, 0].astype(int).ravel()
        else:
            probs_draws = np.asarray(trace["tau_probs"], dtype=float)
            support_draws = np.asarray(trace["tau_support"], dtype=float)
            probs = probs_draws.mean(axis=0)
            support = support_draws[0].astype(int).ravel() if support_draws.ndim > 1 else support_draws.astype(int).ravel()
        probs = probs / probs.sum()
        return support, probs

    raise ValueError("Neither 'tau' nor ('tau_probs' and 'tau_support') found in trace.")


def _tau_mean_samples_from_trace(trace) -> np.ndarray | None:
    """Return per-draw tau means from either tau_mean or tau_probs/tau_support."""
    trace_vars = _available_varnames(trace)
    if "tau_mean" in trace_vars:
        return _values_flat(trace, "tau_mean").astype(float)
    if "tau_probs" in trace_vars and "tau_support" in trace_vars:
        probs = np.stack(_values_by_chain(trace, "tau_probs"), axis=0).astype(float)
        support = np.stack(_values_by_chain(trace, "tau_support"), axis=0).astype(float)
        tau_mean = np.sum(probs * support, axis=-1)
        return tau_mean.reshape(-1)
    return None
def _tau_map_from_trace(trace) -> int:
    """MAP estimate of tau (chunk changepoint index)."""
    support, probs = tau_probabilities(trace)
    return int(support[int(np.argmax(probs))])
def _observed_split_by_tau(
    observed: np.ndarray,
    tau_map: int,
    *,
    likelihood: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split flattened observations into before/after tau regimes by chunk column."""
    obs_2d = np.asarray(observed, dtype=float)
    if obs_2d.ndim == 1:
        obs_2d = obs_2d.reshape(1, -1)
    n_chunks = obs_2d.shape[1]
    chunk_idx = np.arange(n_chunks)
    before_cols = chunk_idx < (int(tau_map) - 1)
    after_cols = ~before_cols
    obs_before = obs_2d[:, before_cols].ravel()
    obs_after = obs_2d[:, after_cols].ravel()
    if likelihood in {"lognormal", "gamma"}:
        obs_before = _positive_feature_values(obs_before)
        obs_after = _positive_feature_values(obs_after)
    else:
        obs_before = obs_before[np.isfinite(obs_before)]
        obs_after = obs_after[np.isfinite(obs_after)]
    return obs_before, obs_after
def p_tau_gt_from_trace(trace, threshold: float = 6.0) -> float:
    """Posterior probability P(tau > threshold)."""
    support, probs = tau_probabilities(trace)
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return float(probs[support > float(threshold)].sum())
def _positive_feature_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    return arr
def _profile_x_grid(observed: np.ndarray, likelihood: str, grid_size: int) -> np.ndarray:
    finite = np.asarray(observed, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Observed feature array has no finite values.")

    if likelihood in {"lognormal", "gamma"}:
        finite = finite[finite > 0.0]
        if finite.size == 0:
            raise ValueError(
                "Observed feature array has no positive values required by positive-only likelihood."
            )
        x_min = max(float(np.min(finite)) * 0.8, 1e-6)
        x_max = float(np.max(finite)) * 1.2
        return np.linspace(x_min, x_max, grid_size)

    if likelihood in {"beta", "interval_inflated_beta"}:
        # Beta / IIB support is strictly (0, 1); use a fixed in-support grid.
        return np.linspace(1e-6, 1.0 - 1e-6, grid_size)

    q1 = float(np.quantile(finite, 0.01))
    q99 = float(np.quantile(finite, 0.99))
    spread = q99 - q1
    if spread <= 0.0:
        spread = max(float(np.std(finite)), 1e-3)
    x_min = q1 - 0.25 * spread
    x_max = q99 + 0.25 * spread
    return np.linspace(x_min, x_max, grid_size)
def _likelihood_pdf_from_posterior(
    *,
    likelihood: str,
    x: np.ndarray,
    params_1: dict,
    params_2: dict,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    eps = 1e-12
    likelihood = str(likelihood).strip().lower()

    if likelihood == "normal":
        mu_1 = float(params_1["mu"])
        mu_2 = float(params_2["mu"])
        sigma_1 = max(float(params_1["sigma"]), eps)
        sigma_2 = max(float(params_2["sigma"]), eps)
        c1 = 1.0 / (sigma_1 * np.sqrt(2.0 * np.pi))
        c2 = 1.0 / (sigma_2 * np.sqrt(2.0 * np.pi))
        y1 = c1 * np.exp(-0.5 * ((x - mu_1) / sigma_1) ** 2)
        y2 = c2 * np.exp(-0.5 * ((x - mu_2) / sigma_2) ** 2)
        return y1, y2

    if likelihood == "student_t":
        nu = max(float(params_1["nu"]), 2.0 + eps)
        mu_1 = float(params_1["mu"])
        mu_2 = float(params_2["mu"])
        sigma_1 = max(float(params_1["sigma"]), eps)
        sigma_2 = max(float(params_2["sigma"]), eps)

        # Student-t PDF via log-space for stability:
        # log f(x) = lgamma((nu+1)/2)-lgamma(nu/2)-0.5*log(nu*pi)-log(sigma)
        #           -((nu+1)/2)*log(1 + ((x-mu)^2)/(nu*sigma^2))
        def _student_t_pdf(xv: np.ndarray, mu: float, sigma: float) -> np.ndarray:
            log_c = (
                float(math.lgamma((nu + 1.0) / 2.0))
                - float(math.lgamma(nu / 2.0))
                - 0.5 * np.log(nu * np.pi)
                - np.log(sigma)
            )
            z2 = ((xv - mu) / sigma) ** 2
            return np.exp(log_c - ((nu + 1.0) / 2.0) * np.log1p(z2 / nu))

        return _student_t_pdf(x, mu_1, sigma_1), _student_t_pdf(x, mu_2, sigma_2)

    if likelihood == "lognormal":
        mu_1 = float(params_1["mu"])
        mu_2 = float(params_2["mu"])
        sigma_1 = max(float(params_1["sigma"]), eps)
        sigma_2 = max(float(params_2["sigma"]), eps)
        x_pos = np.maximum(x, eps)
        c1 = 1.0 / (x_pos * sigma_1 * np.sqrt(2.0 * np.pi))
        c2 = 1.0 / (x_pos * sigma_2 * np.sqrt(2.0 * np.pi))
        y1 = c1 * np.exp(-((np.log(x_pos) - mu_1) ** 2) / (2.0 * sigma_1**2))
        y2 = c2 * np.exp(-((np.log(x_pos) - mu_2) ** 2) / (2.0 * sigma_2**2))
        return y1, y2

    if likelihood == "gamma":
        alpha_1 = max(float(params_1["alpha"]), eps)
        alpha_2 = max(float(params_2["alpha"]), eps)
        beta_1 = max(float(params_1["beta"]), eps)
        beta_2 = max(float(params_2["beta"]), eps)
        x_pos = np.maximum(x, eps)
        y1 = (
            (beta_1**alpha_1)
            / np.exp(float(math.lgamma(alpha_1)))
            * x_pos ** (alpha_1 - 1.0)
            * np.exp(-beta_1 * x_pos)
        )
        y2 = (
            (beta_2**alpha_2)
            / np.exp(float(math.lgamma(alpha_2)))
            * x_pos ** (alpha_2 - 1.0)
            * np.exp(-beta_2 * x_pos)
        )
        return y1, y2

    if likelihood == "beta":
        alpha_1 = max(float(params_1["alpha"]), eps)
        alpha_2 = max(float(params_2["alpha"]), eps)
        beta_1 = max(float(params_1["beta"]), eps)
        beta_2 = max(float(params_2["beta"]), eps)
        x_unit = np.clip(x, eps, 1.0 - eps)
        y1 = (
            np.exp(float(math.lgamma(alpha_1 + beta_1)) - float(math.lgamma(alpha_1)) - float(math.lgamma(beta_1)))
            * (x_unit ** (alpha_1 - 1.0))
            * ((1.0 - x_unit) ** (beta_1 - 1.0))
        )
        y2 = (
            np.exp(float(math.lgamma(alpha_2 + beta_2)) - float(math.lgamma(alpha_2)) - float(math.lgamma(beta_2)))
            * (x_unit ** (alpha_2 - 1.0))
            * ((1.0 - x_unit) ** (beta_2 - 1.0))
        )
        return y1, y2

    if likelihood == "interval_inflated_beta":
        pi_1 = float(params_1.get("pi", 0.0))
        pi_2 = float(params_2.get("pi", 0.0))
        alpha_1 = max(float(params_1["alpha"]), eps)
        alpha_2 = max(float(params_2["alpha"]), eps)
        beta_1 = max(float(params_1["beta"]), eps)
        beta_2 = max(float(params_2["beta"]), eps)
        threshold = float(params_1.get("threshold", params_2.get("threshold", 0.9)))
        x_unit = np.clip(x, eps, 1.0 - eps)

        return (
            interval_inflated_beta_pdf_mixture(x_unit, pi_1, alpha_1, beta_1, threshold),
            interval_inflated_beta_pdf_mixture(x_unit, pi_2, alpha_2, beta_2, threshold),
        )

    raise ValueError(
        f"Unsupported likelihood '{likelihood}'. "
        "Use one of: normal, student_t, lognormal, gamma, beta, interval_inflated_beta."
    )


def _beta_pdf_unit(xv: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Beta(alpha, beta) PDF on (0, 1)."""
    return (
        np.exp(
            float(math.lgamma(alpha + beta))
            - float(math.lgamma(alpha))
            - float(math.lgamma(beta))
        )
        * (xv ** (alpha - 1.0))
        * ((1.0 - xv) ** (beta - 1.0))
    )


def interval_inflated_beta_pdf_mixture(
    xv: np.ndarray,
    pi: float,
    alpha: float,
    beta: float,
    threshold: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Legacy/profile PDF: pi * Uniform[t,1] + (1-pi) * Beta on all x."""
    x_unit = np.clip(np.asarray(xv, dtype=float), eps, 1.0 - eps)
    thr = float(threshold)
    uniform = np.where(x_unit >= thr, 1.0 / max(1.0 - thr, eps), 0.0)
    return float(pi) * uniform + (1.0 - float(pi)) * _beta_pdf_unit(x_unit, alpha, beta)


def interval_inflated_beta_pdf_model(
    xv: np.ndarray,
    pi: float,
    alpha: float,
    beta: float,
    threshold: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Piecewise PDF used by changepoint_model interval_inflated_beta logp.

    - y >= threshold: pi / (1 - threshold)
    - y <  threshold: (1 - pi) * Beta(y)
    """
    x_unit = np.clip(np.asarray(xv, dtype=float), eps, 1.0 - eps)
    thr = float(threshold)
    upper = float(pi) / max(1.0 - thr, eps)
    lower = (1.0 - float(pi)) * _beta_pdf_unit(x_unit, alpha, beta)
    return np.where(x_unit >= thr, upper, lower)


def _likelihood_profile_log_density_y(
    log_density_y: bool | set[tuple[str, str]] | None,
    group_name: str,
    feat_name: str,
) -> bool:
    if log_density_y is True:
        return True
    if not log_density_y:
        return False
    key = (str(group_name).strip().lower(), str(feat_name).strip().lower())
    return key in {
        (str(g).strip().lower(), str(f).strip().lower()) for g, f in log_density_y
    }
def feature_likelihood_profiles(
    trace,
    group_data: dict,
    parameter_selection: dict | None = None,
    *,
    grid_size: int = 300,
    plot: bool = True,
    log_density_y: bool | set[tuple[str, str]] | None = None,
    save_dir: str | os.PathLike[str] | None = None,
    plot_stem_prefix: str = "",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return and optionally plot before/after likelihood profiles for each selected feature.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        Nested dict ``profiles[group_name][feat_name]`` with columns:
        ``x``, ``pdf_before``, ``pdf_after``.
    """
    if grid_size < 50:
        raise ValueError("grid_size must be >= 50.")

    trace_vars = _available_varnames(trace)
    active_features = {feat for features in group_data.values() for feat in features.keys()}
    parameter_cfg = _parse_parameter_selection(parameter_selection, active_features)
    profiles: dict[str, dict[str, pd.DataFrame]] = {}

    for group_name, features in group_data.items():
        profiles[group_name] = {}
        for feat_name, observed_df in features.items():
            likelihood = str(parameter_cfg[feat_name].get("likelihood", "normal")).strip().lower()
            observed_2d = observed_df.to_numpy(dtype=float)
            observed = observed_2d.reshape(-1)

            x = _profile_x_grid(observed, likelihood=likelihood, grid_size=grid_size)
            params_1: dict[str, float] = {}
            params_2: dict[str, float] = {}

            for p in ("mu", "sigma", "alpha", "beta", "pi"):
                p1 = f"{p}_{group_name}_{feat_name}_1"
                p2 = f"{p}_{group_name}_{feat_name}_2"
                if p1 in trace_vars and p2 in trace_vars:
                    params_1[p] = float(np.mean(_values_flat(trace, p1)))
                    params_2[p] = float(np.mean(_values_flat(trace, p2)))

            if likelihood == "interval_inflated_beta":
                threshold = float(parameter_cfg[feat_name].get("threshold", 0.9))
                params_1["threshold"] = threshold
                params_2["threshold"] = threshold

            nu_name = f"nu_{group_name}_{feat_name}"
            if nu_name in trace_vars:
                nu_mean = float(np.mean(_values_flat(trace, nu_name)))
                params_1["nu"] = nu_mean
                params_2["nu"] = nu_mean

            y_before, y_after = _likelihood_pdf_from_posterior(
                likelihood=likelihood,
                x=x,
                params_1=params_1,
                params_2=params_2,
            )
            profile_cols: dict[str, np.ndarray] = {
                "x": x,
                "pdf_before": y_before,
                "pdf_after": y_after,
            }
            if likelihood == "interval_inflated_beta":
                threshold = float(params_1["threshold"])
                profile_cols["pdf_before_model"] = interval_inflated_beta_pdf_model(
                    x, params_1["pi"], params_1["alpha"], params_1["beta"], threshold
                )
                profile_cols["pdf_after_model"] = interval_inflated_beta_pdf_model(
                    x, params_2["pi"], params_2["alpha"], params_2["beta"], threshold
                )
                profile_cols["pdf_before_mixture"] = y_before
                profile_cols["pdf_after_mixture"] = y_after
            profile_df = pd.DataFrame(profile_cols)
            profiles[group_name][feat_name] = profile_df

            if plot:
                use_log_y = _likelihood_profile_log_density_y(
                    log_density_y, group_name, feat_name
                )
                from pathlib import Path

                from seismic_pipeline.visualization.changepoint_plots import (
                    plot_feature_likelihood_profile,
                    plot_interval_inflated_beta_pdf_comparison,
                    plot_observations_before_after,
                )

                lik_save = obs_save = iib_cmp_save = None
                if save_dir is not None:
                    stem = f"{plot_stem_prefix}_" if plot_stem_prefix else ""
                    lik_save = Path(save_dir) / f"{stem}{group_name}_{feat_name}_likelihood.png"
                    obs_save = Path(save_dir) / f"{stem}{group_name}_{feat_name}_observations.png"
                    if likelihood == "interval_inflated_beta":
                        iib_cmp_save = (
                            Path(save_dir)
                            / f"{stem}{group_name}_{feat_name}_iib_pdf_compare.png"
                        )

                plot_feature_likelihood_profile(
                    group_name=group_name,
                    feat_name=feat_name,
                    likelihood=likelihood,
                    x=x,
                    y_before=y_before,
                    y_after=y_after,
                    observed_2d=observed_2d,
                    trace=trace,
                    use_log_y=use_log_y,
                    save_path=lik_save,
                )
                if likelihood == "interval_inflated_beta":
                    plot_interval_inflated_beta_pdf_comparison(
                        group_name=group_name,
                        feat_name=feat_name,
                        x=x,
                        y_before_model=profile_df["pdf_before_model"].to_numpy(dtype=float),
                        y_after_model=profile_df["pdf_after_model"].to_numpy(dtype=float),
                        y_before_mixture=profile_df["pdf_before_mixture"].to_numpy(dtype=float),
                        y_after_mixture=profile_df["pdf_after_mixture"].to_numpy(dtype=float),
                        threshold=float(params_1["threshold"]),
                        use_log_y=use_log_y,
                        save_path=iib_cmp_save,
                    )
                if obs_save is not None:
                    plot_observations_before_after(
                        group_name=group_name,
                        feat_name=feat_name,
                        likelihood=likelihood,
                        observed_2d=observed_2d,
                        trace=trace,
                        title_prefix=plot_stem_prefix,
                        save_path=obs_save,
                    )

    return profiles
def changepoint_model_config_fingerprint(config: dict) -> str:
    """Stable hash for caching / deduplication of discrete model configurations."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
def _posterior_dict_from_trace(trace) -> dict[str, np.ndarray]:
    """Stack posterior samples as (chain, draw, *shape) for ArviZ."""
    names = sorted(_available_varnames(trace))
    posterior: dict[str, np.ndarray] = {}
    for name in names:
        try:
            chains = _values_by_chain(trace, name)
            posterior[name] = np.stack(chains, axis=0)
        except Exception:
            continue
    return posterior
def _float_ic_scalar(val: Any) -> float:
    try:
        if val is None:
            return float("nan")
        return float(np.asarray(val, dtype=float).squeeze())
    except Exception:
        return float("nan")
def _has_log_likelihood_group(trace: DataTree) -> bool:
    return "log_likelihood" in trace.children and bool(trace["log_likelihood"].data_vars)
def _ic_elpd(ic_result: Any) -> float:
    for attr in ("elpd_loo", "elpd_waic", "elpd"):
        if hasattr(ic_result, attr):
            val = getattr(ic_result, attr)
            if val is not None:
                return _float_ic_scalar(val)
    return float("nan")
def _ic_p(ic_result: Any) -> float:
    for attr in ("p_loo", "p_waic", "p"):
        if hasattr(ic_result, attr):
            val = getattr(ic_result, attr)
            if val is not None:
                return _float_ic_scalar(val)
    return float("nan")
def _call_az_waic(idata_ic: DataTree):
    waic_fn = getattr(az, "waic", None)
    if waic_fn is None:
        return None
    try:
        return waic_fn(idata_ic, scale="log")
    except TypeError:
        return waic_fn(idata_ic)


def idata_for_waic_from_trace(trace, model) -> DataTree:
    """Build DataTree with a log_likelihood group suitable for ``az.loo``."""
    if _is_inferencedata(trace) and _has_log_likelihood_group(trace):
        return trace

    trace_vars = _available_varnames(trace)

    # Prefer precomputed marginalized pointwise log-lik from the trace (fast, correct
    # for tau_mode='marginalized'). Do NOT call pm.compute_log_likelihood first: it
    # re-scans every posterior draw via PyTensor (~100x slower) and may not populate
    # log_likelihood for Potential-only observation models.
    if "changepoint_pointwise_log_lik" in trace_vars:
        return _attach_loglik_idata_from_posterior(trace, "changepoint_pointwise_log_lik")

    if "changepoint_joint_log_lik" in trace_vars:
        return _attach_loglik_idata_from_posterior(trace, "changepoint_joint_log_lik")

    posterior = _posterior_dict_from_trace(trace)
    if not posterior:
        raise ValueError("No posterior variables found in trace for WAIC/LOO.")

    if _is_inferencedata(trace):
        try:
            with model:
                idata = pm.compute_log_likelihood(trace, model=model)
            if _has_log_likelihood_group(idata):
                return idata
        except Exception:
            pass

    idata = az.from_dict({"posterior": posterior})
    try:
        with model:
            idata = pm.compute_log_likelihood(idata, model=model)
    except Exception as exc:
        raise RuntimeError(
            "WAIC/LOO requires pointwise log-likelihood; compute_log_likelihood failed. "
            "For marginalized tau models use changepoint_joint_log_lik in the trace."
        ) from exc
    if not _has_log_likelihood_group(idata):
        raise RuntimeError("compute_log_likelihood did not populate log_likelihood.")
    return idata
_idata_for_waic_from_trace = idata_for_waic_from_trace
def score_changepoint_trace(
    trace,
    *,
    group_data: dict,
    parameter_selection: dict | None,
    tau_threshold: float = 7.0,
    summary_var_names: List[str] | None = None,
    model=None,
    criterion: str = "waic",
    warn_on_fallback: bool = True,
    loo_report: str = "ic",
) -> dict[str, Any]:
    """Summarize a changepoint trace for model comparison and Metropolis-Hastings scoring.

    Returns keys including: p_tau_gt_threshold, map_tau, map_tau_prob, tau_entropy,
    tau_concentration, tau_q1, tau_q2, tau_q3, tau_hdi_60_lower, tau_hdi_60_upper,
    tau_hdi_60_width, r_hat_max, ess_min_bulk, ess_min_tail, n_divergences, bfmi / bfmi_approx,
    and when ``model`` is provided: elpd, waic / loo, p_waic / p_loo, criterion metadata.

    ``loo_report`` controls the scale stored in ``loo``: ``"elpd"`` uses ArviZ ``elpd_loo``
    (log predictive density); ``"ic"`` uses LOO-IC (``-2 * elpd_loo`` when ``.loo`` is absent).
    """
    support, probs = tau_probabilities(trace)
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    p_gt = float(probs[support > float(tau_threshold)].sum())
    e_tau = float(np.sum(support * probs))
    e_tau2 = float(np.sum((support ** 2) * probs))
    tau_std = float(np.sqrt(max(e_tau2 - e_tau**2, 0.0)))
    map_idx = int(np.argmax(probs))
    map_tau = int(support[map_idx])
    map_p = float(probs[map_idx])
    ent = float(-np.sum(probs * np.log(probs + 1e-20)))
    max_ent = float(np.log(max(len(probs), 1)))
    conc = float(1.0 - ent / max_ent) if max_ent > 1e-9 else 0.0

    tau_q1 = float("nan")
    tau_q2 = float("nan")
    tau_q3 = float("nan")
    tau_hdi_60_lower = float("nan")
    tau_hdi_60_upper = float("nan")
    tau_hdi_60_width = float("nan")
    tau_mean_samples = _tau_mean_samples_from_trace(trace)
    if tau_mean_samples is not None:
        tau_q1 = float(np.percentile(tau_mean_samples, 25))
        tau_q2 = float(np.percentile(tau_mean_samples, 50))
        tau_q3 = float(np.percentile(tau_mean_samples, 75))
        hdi_60 = az.hdi(tau_mean_samples, prob=0.60)
        tau_hdi_60_lower = float(hdi_60[0])
        tau_hdi_60_upper = float(hdi_60[1])
        tau_hdi_60_width = tau_hdi_60_upper - tau_hdi_60_lower
    elif warn_on_fallback:
        warnings.warn(
            "score_changepoint_trace: neither ``tau_mean`` nor ``tau_probs/tau_support`` in trace; "
            "tau_q1/q2/q3 and tau_hdi_60_* set to NaN.",
            UserWarning,
            stacklevel=2,
        )

    r_hat_max = 1.0
    ess_min_bulk = float("inf")
    ess_min_tail = float("inf")
    if summary_var_names:
        try:
            summ = summary_from_trace(trace, summary_var_names)
            if "r_hat" in summ.columns:
                rh = summ["r_hat"].to_numpy(dtype=float)
                rh = rh[np.isfinite(rh)]
                if rh.size:
                    r_hat_max = float(np.nanmax(rh))
            if "ess_bulk" in summ.columns:
                eb = summ["ess_bulk"].to_numpy(dtype=float)
                eb = eb[np.isfinite(eb)]
                if eb.size:
                    ess_min_bulk = float(np.nanmin(eb))
            if "ess_tail" in summ.columns:
                et = summ["ess_tail"].to_numpy(dtype=float)
                et = et[np.isfinite(et)]
                if et.size:
                    ess_min_tail = float(np.nanmin(et))
        except Exception:
            pass

    n_div = 0
    try:
        diverging = _sampler_stat(trace, "diverging")
        n_div = int(np.asarray(diverging).sum())
    except Exception:
        pass

    bfmi = float("nan")
    try:
        bfmi_arr = np.asarray(az.bfmi(trace), dtype=float).reshape(-1)
        bfmi_arr = bfmi_arr[np.isfinite(bfmi_arr)]
        if bfmi_arr.size:
            bfmi = float(np.mean(bfmi_arr))
    except Exception:
        try:
            energy = np.asarray(_sampler_stat(trace, "energy"), dtype=float).reshape(-1)
            if energy.size > 1 and np.var(energy) > 0:
                bfmi = float(np.mean(np.diff(energy) ** 2) / np.var(energy))
        except Exception:
            pass

    n_feat_blocks = sum(len(v) for v in group_data.values())
    n_events = 0
    n_chunks = 0
    if group_data:
        first_group = next(iter(group_data.values()), {})
        if first_group:
            first_block = next(iter(first_group.values()))
            n_events = int(first_block.shape[0])
            n_chunks = int(first_block.shape[1])
    n_observations = int(n_events * n_chunks)
    active_features = sorted({feat for feats in group_data.values() for feat in feats.keys()})
    likelihoods: dict[str, str] = {}
    if parameter_selection:
        for f in active_features:
            if f in parameter_selection and isinstance(parameter_selection[f], dict):
                likelihoods[f] = str(parameter_selection[f].get("likelihood", "")).lower()

    crit = str(criterion).strip().lower()
    elpd = float("nan")
    waic_stat = float("nan")
    p_waic = float("nan")
    loo_stat = float("nan")
    loo_ic = float("nan")
    elpd_loo = float("nan")
    p_loo = float("nan")
    elpd_waic = float("nan")
    waic_warning_flag = False
    waic_warning_messages: List[str] = []
    criterion_error: str | None = None
    ic_computed = False
    idata_ic_cache: DataTree | None = None
    loo_obj_cache: Any | None = None

    if model is not None and crit in {"waic", "loo"}:
        try:
            idata_ic = _idata_for_waic_from_trace(trace, model)
            idata_ic_cache = idata_ic
            if crit == "waic" and getattr(az, "waic", None) is not None:
                with warnings.catch_warnings(record=True) as waic_warns:
                    warnings.simplefilter("always")
                    ic_waic = _call_az_waic(idata_ic)
                if ic_waic is not None:
                    p_waic = _ic_p(ic_waic)
                    waic_stat = _float_ic_scalar(getattr(ic_waic, "waic", float("nan")))
                    elpd_waic = _ic_elpd(ic_waic)
                    if not math.isfinite(waic_stat) and math.isfinite(elpd_waic):
                        waic_stat = float(-2.0 * elpd_waic)
                    for w in waic_warns:
                        msg = str(w.message)
                        waic_warning_messages.append(msg)
                        if "posterior variance of the log predictive densities exceeds 0.4" in msg:
                            waic_warning_flag = True

            ic_loo = None
            if crit == "loo" or crit == "waic":
                ic_loo = az.loo(idata_ic, pointwise=True)
                loo_obj_cache = ic_loo
                p_loo = _ic_p(ic_loo)
                elpd_loo = _ic_elpd(ic_loo)
                loo_ic = _float_ic_scalar(getattr(ic_loo, "loo", float("nan")))
                if not math.isfinite(loo_ic) and math.isfinite(elpd_loo):
                    loo_ic = float(-2.0 * elpd_loo)
                if str(loo_report).strip().lower() == "elpd":
                    loo_stat = elpd_loo
                else:
                    loo_stat = loo_ic

            if crit == "waic":
                elpd = elpd_waic
            elif crit == "loo" and ic_loo is not None:
                elpd = elpd_loo
            ic_computed = True
        except Exception as exc:
            criterion_error = str(exc)
            elpd = float("-inf")
            ic_computed = True
    elif model is None and warn_on_fallback:
        warnings.warn(
            "score_changepoint_trace: ``model`` is None; skipping WAIC/LOO. "
            "Metropolis-Hastings will use the legacy tau-based log-target unless you pass ``model``.",
            UserWarning,
            stacklevel=2,
        )

    out: dict[str, Any] = {
        "p_tau_gt_threshold": p_gt,
        "e_tau": e_tau,
        "tau_std": tau_std,
        "map_tau": map_tau,
        "map_tau_prob": map_p,
        "tau_entropy": ent,
        "tau_concentration": conc,
        "tau_q1": tau_q1,
        "tau_q2": tau_q2,
        "tau_q3": tau_q3,
        "tau_hdi_60_lower": tau_hdi_60_lower,
        "tau_hdi_60_upper": tau_hdi_60_upper,
        "tau_hdi_60_width": tau_hdi_60_width,
        "r_hat_max": r_hat_max,
        "ess_min_bulk": ess_min_bulk,
        "ess_min_tail": ess_min_tail,
        "n_divergences": n_div,
        "bfmi": bfmi,
        "bfmi_approx": bfmi,
        "n_feature_blocks": n_feat_blocks,
        "n_events": n_events,
        "n_chunks": n_chunks,
        "n_observations": n_observations,
        "active_features": active_features,
        "likelihoods": likelihoods,
        "elpd": elpd,
        "criterion": crit if model is not None else "none",
        "ic_computed": ic_computed,
        "waic": waic_stat,
        "p_waic": p_waic,
        "elpd_loo": elpd_loo,
        "loo_ic": loo_ic,
        "loo_reported": loo_stat,
        "loo": loo_stat,
        "p_loo": p_loo,
        "waic_warning_flag": waic_warning_flag,
        "waic_warning_messages": waic_warning_messages,
        "criterion_error": criterion_error,
        "_idata_ic": idata_ic_cache,
        "_loo_obj": loo_obj_cache,
    }
    return out
def changepoint_log_target(
    score_parts: dict[str, Any],
    *,
    w_elpd: float = 1.0,
    r_hat_gate: float = 1.05,
    ess_threshold: float = 100.0,
    bfmi_threshold: float = 0.3,
    w_p_tau: float = 0.2,
    w_map: float = 0.05,
    w_conc: float = 0.05,
    w_complexity: float = 0.05,
    w_rhat_penalty: float = 40.0,
    w_bfmi_penalty: float = 25.0,
    w_ess_penalty: float = 0.15,
    r_hat_gate_legacy: float = 1.01,
    w_p_tau_legacy: float = 8.0,
    w_map_legacy: float = 2.0,
    w_conc_legacy: float = 1.0,
    w_ess_legacy: float = 0.002,
    w_complexity_legacy: float = 0.08,
) -> float:
    """Log-scale score for Metropolis-Hastings; higher is better.

    When ``score_parts['ic_computed']`` is true (WAIC/LOO ran), **elpd** dominates; small
    bonuses use :math:`P(\\tau > \\text{thr})`, MAP :math:`\\tau` mass, and concentration.
    Otherwise the legacy tau- and ESS-weighted score is used (backward compatible).
    """
    p = float(score_parts.get("p_tau_gt_threshold", 0.0))
    mp = float(score_parts.get("map_tau_prob", 0.0))
    conc = float(score_parts.get("tau_concentration", 0.0))
    rmax = float(score_parts.get("r_hat_max", 1.0))
    essb = float(score_parts.get("ess_min_bulk", float("inf")))
    esst = float(score_parts.get("ess_min_tail", float("inf")))
    ndiv = int(score_parts.get("n_divergences", 0))
    nblk = int(score_parts.get("n_feature_blocks", 1))
    bfmi = float(score_parts.get("bfmi", score_parts.get("bfmi_approx", float("nan"))))

    if ndiv > 0:
        return float("-inf")

    ic_computed = bool(score_parts.get("ic_computed", False))
    logp = math.log(max(p, 1e-12))

    if ic_computed:
        elpd = float(score_parts.get("elpd", float("nan")))
        if not math.isfinite(elpd):
            return float("-inf")

        pen = 0.0
        if rmax > r_hat_gate:
            pen -= w_rhat_penalty * (rmax - r_hat_gate)

        ess_min = float("inf")
        if math.isfinite(essb):
            ess_min = min(ess_min, essb)
        if math.isfinite(esst):
            ess_min = min(ess_min, esst)
        if not math.isfinite(ess_min) or ess_min < ess_threshold:
            short = ess_threshold if not math.isfinite(ess_min) else max(0.0, ess_threshold - ess_min)
            pen -= w_ess_penalty * short

        if math.isfinite(bfmi) and bfmi < bfmi_threshold:
            pen -= w_bfmi_penalty * (bfmi_threshold - bfmi)

        return (
            w_elpd * elpd
            + w_p_tau * logp
            + w_map * math.log(max(mp, 1e-12))
            + w_conc * conc
            - w_complexity * float(max(nblk - 1, 0))
            + pen
        )

    gate = 0.0
    if rmax > r_hat_gate_legacy:
        gate -= 50.0 * (rmax - r_hat_gate_legacy)
    if not math.isfinite(essb) or essb < 100.0:
        gate -= 5.0
    ess_term = math.log(max(essb, 1.0)) if math.isfinite(essb) else 0.0

    return (
        w_p_tau_legacy * logp
        + w_map_legacy * math.log(max(mp, 1e-12))
        + w_conc_legacy * conc
        + w_ess_legacy * ess_term
        - w_complexity_legacy * float(max(nblk - 1, 0))
        + gate
    )
def collect_pareto_k_stats(
    trace,
    model,
    *,
    pareto_threshold: float = 0.7,
    idata_ic: DataTree | None = None,
    loo_obj: Any | None = None,
) -> tuple[float, int, list[int], int | None]:
    try:
        if loo_obj is None:
            if idata_ic is None:
                idata_ic = idata_for_waic_from_trace(trace, model)
            loo_obj = az.loo(idata_ic, pointwise=True)
        pareto = getattr(loo_obj, "pareto_k", None)
        if pareto is None:
            return float("nan"), 0, [], None
        pareto_raw = np.asarray(pareto, dtype=float).reshape(-1)
        if pareto_raw.size == 0:
            return float("nan"), 0, [], None
        finite_mask = np.isfinite(pareto_raw)
        if not bool(np.any(finite_mask)):
            return float("nan"), 0, [], None
        pareto_vals = pareto_raw[finite_mask]
        over_mask = finite_mask & (pareto_raw > float(pareto_threshold))
        over_indices = np.flatnonzero(over_mask).astype(int).tolist()
        worst_local_idx = int(np.argmax(np.where(finite_mask, pareto_raw, float("-inf"))))
        return float(np.max(pareto_vals)), int(len(over_indices)), over_indices, worst_local_idx
    except Exception:
        return float("nan"), 0, [], None
_collect_pareto_k_stats = collect_pareto_k_stats
