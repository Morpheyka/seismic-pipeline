"""Posterior predictive check plotting for changepoint models."""
from __future__ import annotations

import warnings
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from seismic_pipeline.bayesian.diagnostics import _available_varnames, _is_inferencedata, _values_flat
from seismic_pipeline.bayesian.priors import _parse_parameter_selection
from seismic_pipeline.bayesian.changepoint_model import sample_interval_inflated_beta


def _posterior_stack_chains_draws(trace, var_name: str) -> np.ndarray:
    """Stack posterior samples as (chain, draw, ...)."""
    if _is_inferencedata(trace):
        return np.asarray(trace.posterior[var_name])
    return np.stack(trace.get_values(var_name, combine=False), axis=0)


def _marginalized_changepoint_ppc_first_feature(
    trace,
    group_data: dict,
    parameter_selection: dict | None,
    *,
    rng: np.random.Generator,
    num_pp_samples: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Simulate replicated data for marginalized-tau changepoint models."""
    if "tau_probs" not in _available_varnames(trace):
        raise KeyError("tau_probs")
    active_features = {feat for feats in group_data.values() for feat in feats.keys()}
    param_cfg = _parse_parameter_selection(parameter_selection, active_features)

    first_group = next(iter(group_data.keys()))
    first_feat = next(iter(group_data[first_group].keys()))
    label = f"obs_{first_group}_{first_feat}"
    y_obs = np.asarray(group_data[first_group][first_feat].to_numpy(), dtype=float)
    n_rows, n_chunks = y_obs.shape
    spec = param_cfg[first_feat]
    likelihood = str(spec.get("likelihood", "normal")).strip().lower()

    tau_probs = _posterior_stack_chains_draws(trace, "tau_probs")
    tau_support = _posterior_stack_chains_draws(trace, "tau_support")
    c, d, k_tau = tau_probs.shape
    if tau_support.shape != tau_probs.shape:
        raise ValueError("tau_support and tau_probs must have the same shape in the trace.")
    ts0 = np.asarray(tau_support[0, 0, :], dtype=np.int64)

    tp = tau_probs.reshape(-1, k_tau)
    tp = np.clip(tp, 1e-15, np.inf)
    tp /= tp.sum(axis=1, keepdims=True)
    n_flat = c * d
    idx_flat = np.arange(n_flat)
    if n_flat > int(num_pp_samples):
        idx_flat = rng.choice(n_flat, size=int(num_pp_samples), replace=False)
    s = int(idx_flat.size)

    y_pp = np.empty((s, n_rows, n_chunks), dtype=float)

    def _scalar_at(ci: int, di: int, name: str) -> float:
        return float(_posterior_stack_chains_draws(trace, name)[ci, di])

    for ii, flat_i in enumerate(idx_flat):
        ci, di = divmod(int(flat_i), d)
        probs = tp[int(flat_i)]
        k = int(rng.choice(k_tau, p=probs))
        tau_val = int(ts0[k])

        mu1 = _scalar_at(ci, di, f"mu_{first_group}_{first_feat}_1")
        mu2 = _scalar_at(ci, di, f"mu_{first_group}_{first_feat}_2")
        s1 = _scalar_at(ci, di, f"sigma_{first_group}_{first_feat}_1")
        s2 = _scalar_at(ci, di, f"sigma_{first_group}_{first_feat}_2")

        for j in range(n_chunks):
            use_r1 = tau_val > (j + 1)
            if likelihood == "normal":
                mu, sig = (mu1, s1) if use_r1 else (mu2, s2)
                y_pp[ii, :, j] = rng.normal(mu, sig, size=n_rows)
            elif likelihood == "student_t":
                nu = _scalar_at(ci, di, f"nu_{first_group}_{first_feat}")
                mu, sig = (mu1, s1) if use_r1 else (mu2, s2)
                y_pp[ii, :, j] = mu + sig * rng.standard_t(df=nu, size=n_rows)
            elif likelihood == "lognormal":
                mu, sig = (mu1, s1) if use_r1 else (mu2, s2)
                y_pp[ii, :, j] = rng.lognormal(mean=mu, sigma=sig, size=n_rows)
            elif likelihood == "gamma":
                a1 = _scalar_at(ci, di, f"alpha_{first_group}_{first_feat}_1")
                a2 = _scalar_at(ci, di, f"alpha_{first_group}_{first_feat}_2")
                b1 = _scalar_at(ci, di, f"beta_{first_group}_{first_feat}_1")
                b2 = _scalar_at(ci, di, f"beta_{first_group}_{first_feat}_2")
                alpha, beta = ((a1, b1) if use_r1 else (a2, b2))
                y_pp[ii, :, j] = rng.gamma(shape=alpha, scale=1.0 / beta, size=n_rows)
            elif likelihood == "beta":
                a1 = _scalar_at(ci, di, f"alpha_{first_group}_{first_feat}_1")
                a2 = _scalar_at(ci, di, f"alpha_{first_group}_{first_feat}_2")
                b1 = _scalar_at(ci, di, f"beta_{first_group}_{first_feat}_1")
                b2 = _scalar_at(ci, di, f"beta_{first_group}_{first_feat}_2")
                alpha, beta = ((a1, b1) if use_r1 else (a2, b2))
                y_pp[ii, :, j] = rng.beta(alpha, beta, size=n_rows)
            elif likelihood == "interval_inflated_beta":
                pi1 = _scalar_at(ci, di, f"pi_{first_group}_{first_feat}_1")
                pi2 = _scalar_at(ci, di, f"pi_{first_group}_{first_feat}_2")
                a1 = _scalar_at(ci, di, f"alpha_{first_group}_{first_feat}_1")
                a2 = _scalar_at(ci, di, f"alpha_{first_group}_{first_feat}_2")
                b1 = _scalar_at(ci, di, f"beta_{first_group}_{first_feat}_1")
                b2 = _scalar_at(ci, di, f"beta_{first_group}_{first_feat}_2")
                pi, alpha, beta = ((pi1, a1, b1) if use_r1 else (pi2, a2, b2))
                thr = float(spec.get("threshold", 0.9))
                y_pp[ii, :, j] = sample_interval_inflated_beta(
                    rng, n_rows, pi, alpha, beta, thr
                )
            else:
                raise ValueError(
                    f"Unsupported likelihood '{likelihood}' for marginalized PPC "
                    f"(feature '{first_feat}')."
                )

    return y_pp, y_obs, label


def _discrete_changepoint_ppc_first_feature(
    trace,
    group_data: dict,
    parameter_selection: dict | None,
    *,
    rng: np.random.Generator,
    num_pp_samples: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Simulate replicated data for discrete-tau changepoint models (Potential-based likelihoods)."""
    if "tau" not in _available_varnames(trace):
        raise KeyError("tau")
    active_features = {feat for feats in group_data.values() for feat in feats.keys()}
    param_cfg = _parse_parameter_selection(parameter_selection, active_features)

    first_group = next(iter(group_data.keys()))
    first_feat = next(iter(group_data[first_group].keys()))
    label = f"obs_{first_group}_{first_feat}"
    y_obs = np.asarray(group_data[first_group][first_feat].to_numpy(), dtype=float)
    n_rows, n_chunks = y_obs.shape
    spec = param_cfg[first_feat]
    likelihood = str(spec.get("likelihood", "normal")).strip().lower()

    tau_draws = _values_flat(trace, "tau").astype(int)
    n_flat = tau_draws.size
    if n_flat > int(num_pp_samples):
        idx_flat = rng.choice(n_flat, size=int(num_pp_samples), replace=False)
    else:
        idx_flat = np.arange(n_flat)
    s = int(idx_flat.size)

    y_pp = np.empty((s, n_rows, n_chunks), dtype=float)

    def _scalar_at(flat_i: int, name: str) -> float:
        return float(_values_flat(trace, name)[flat_i])

    for ii, flat_i in enumerate(idx_flat):
        tau_val = int(tau_draws[int(flat_i)])
        mu1 = _scalar_at(int(flat_i), f"mu_{first_group}_{first_feat}_1")
        mu2 = _scalar_at(int(flat_i), f"mu_{first_group}_{first_feat}_2")
        s1 = _scalar_at(int(flat_i), f"sigma_{first_group}_{first_feat}_1")
        s2 = _scalar_at(int(flat_i), f"sigma_{first_group}_{first_feat}_2")

        for j in range(n_chunks):
            use_r1 = tau_val > (j + 1)
            if likelihood == "interval_inflated_beta":
                pi1 = _scalar_at(int(flat_i), f"pi_{first_group}_{first_feat}_1")
                pi2 = _scalar_at(int(flat_i), f"pi_{first_group}_{first_feat}_2")
                a1 = _scalar_at(int(flat_i), f"alpha_{first_group}_{first_feat}_1")
                a2 = _scalar_at(int(flat_i), f"alpha_{first_group}_{first_feat}_2")
                b1 = _scalar_at(int(flat_i), f"beta_{first_group}_{first_feat}_1")
                b2 = _scalar_at(int(flat_i), f"beta_{first_group}_{first_feat}_2")
                pi, alpha, beta = ((pi1, a1, b1) if use_r1 else (pi2, a2, b2))
                thr = float(spec.get("threshold", 0.9))
                y_pp[ii, :, j] = sample_interval_inflated_beta(
                    rng, n_rows, pi, alpha, beta, thr
                )
            else:
                raise ValueError(
                    f"Unsupported likelihood '{likelihood}' for discrete-tau manual PPC "
                    f"(feature '{first_feat}')."
                )

    return y_pp, y_obs, label


def _ppc_sample_ndim(y_pp: np.ndarray, y_obs: np.ndarray | None = None) -> int:
    y_pp = np.asarray(y_pp)
    if y_obs is not None:
        y_obs_arr = np.asarray(y_obs)
        n_obs = int(np.prod(y_obs_arr.shape)) if y_obs_arr.size else 0
        if n_obs > 0:
            for sample_ndim in range(1, y_pp.ndim):
                if int(np.prod(y_pp.shape[sample_ndim:])) == n_obs:
                    return sample_ndim
    return 2 if y_pp.ndim >= 4 else 1


def _flatten_ppc_draws_and_obs(
    y_pp: np.ndarray,
    y_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y_pp = np.asarray(y_pp, dtype=float)
    if y_pp.ndim < 2:
        y_pp = y_pp.reshape(-1, 1)
    sample_ndim = _ppc_sample_ndim(y_pp, y_obs)
    y_samples = y_pp.reshape((-1,) + y_pp.shape[sample_ndim:])
    n_draws = y_samples.shape[0]
    y_flat = y_samples.reshape(n_draws, -1)
    pred_mean = np.asarray(y_pp, dtype=float).mean(axis=tuple(range(sample_ndim))).reshape(-1)
    y_obs_flat = None
    if y_obs is not None:
        y_obs_flat = np.asarray(y_obs, dtype=float).reshape(-1)
        if y_obs_flat.size != pred_mean.size:
            y_obs_flat = None
    return y_flat, pred_mean, y_obs_flat


def _observed_for_ppc_var(
    observed_data: np.ndarray | dict[str, np.ndarray] | None,
    obs_rvs,
    var_name: str,
    index: int,
) -> np.ndarray | None:
    if isinstance(observed_data, dict):
        if var_name in observed_data:
            return np.asarray(observed_data[var_name], dtype=float)
        return None
    if observed_data is not None and index == 0:
        return np.asarray(observed_data, dtype=float)
    if obs_rvs and index < len(obs_rvs):
        try:
            return np.asarray(obs_rvs[index].eval(), dtype=float)
        except Exception:
            return None
    return None


def _plot_ppc_flattened_on_ax(
    ax,
    y_pp: np.ndarray,
    y_obs: np.ndarray | None = None,
    *,
    title: str | None = None,
) -> None:
    y_flat, _pred_mean, y_obs_flat = _flatten_ppc_draws_and_obs(y_pp, y_obs)
    q05 = np.quantile(y_flat, 0.05, axis=0)
    q50 = np.quantile(y_flat, 0.50, axis=0)
    q95 = np.quantile(y_flat, 0.95, axis=0)
    x = np.arange(q50.shape[0], dtype=int)
    ax.fill_between(x, q05, q95, color="#1f77b4", alpha=0.22, label="posterior predictive 90% band")
    ax.plot(x, q50, color="#1f77b4", lw=1.6, label="posterior predictive median")
    if y_obs_flat is not None:
        ax.plot(x, y_obs_flat, color="#E24A33", lw=1.4, alpha=0.9, label="observed")
    ax.set_xlabel("observation index")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def plot_posterior_predictive_check(
    trace,
    model,
    observed_data: np.ndarray | dict[str, np.ndarray] | None = None,
    *,
    group_data: dict | None = None,
    parameter_selection: dict | None = None,
    num_pp_samples: int = 300,
    random_seed: int | None = None,
) -> Any:
    """Posterior predictive check for changepoint models."""
    rng = np.random.default_rng(random_seed)
    obs_rvs = getattr(model, "observed_RVs", None) or []
    obs_var_names = [rv.name for rv in obs_rvs]
    ppc = None
    y_pp = None
    y_obs = observed_data

    if obs_var_names:
        with model:
            ppc = pm.sample_posterior_predictive(
                trace,
                var_names=obs_var_names,
                random_seed=random_seed,
                return_inferencedata=True,
                extend_inferencedata=False,
                predictions=False,
            )

        try:
            y_pp = np.asarray(ppc.posterior_predictive[obs_var_names[0]], dtype=float)
        except Exception:
            y_pp = None

        if y_obs is None and obs_rvs:
            try:
                y_obs = np.asarray(obs_rvs[0].eval(), dtype=float)
            except Exception:
                y_obs = None

        if y_pp is None:
            az.plot_ppc(ppc, num_pp_samples=int(max(20, num_pp_samples)))
            plt.tight_layout()
            plt.show()
            return ppc
    elif "tau_probs" in _available_varnames(trace) and group_data is not None:
        try:
            y_pp, y_obs_default, _label = _marginalized_changepoint_ppc_first_feature(
                trace,
                group_data,
                parameter_selection,
                rng=rng,
                num_pp_samples=num_pp_samples,
            )
        except Exception as exc:
            warnings.warn(
                "plot_posterior_predictive_check: marginalized tau model but PPC simulation failed "
                f"({exc}).",
                UserWarning,
                stacklevel=2,
            )
            return None
        y_obs = observed_data if observed_data is not None else y_obs_default
    elif "tau" in _available_varnames(trace) and group_data is not None:
        try:
            y_pp, y_obs_default, _label = _discrete_changepoint_ppc_first_feature(
                trace,
                group_data,
                parameter_selection,
                rng=rng,
                num_pp_samples=num_pp_samples,
            )
        except Exception as exc:
            warnings.warn(
                "plot_posterior_predictive_check: discrete tau model but manual PPC simulation failed "
                f"({exc}).",
                UserWarning,
                stacklevel=2,
            )
            return None
        y_obs = observed_data if observed_data is not None else y_obs_default
    else:
        if not obs_var_names:
            warnings.warn(
                "plot_posterior_predictive_check: no PyMC observed RVs (e.g. marginalized tau). "
                "Pass group_data from prepare_model_data / MH cache to enable PPC, or use "
                "tau_mode='discrete'.",
                UserWarning,
                stacklevel=2,
            )
        return None

    panels: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    if ppc is not None and len(obs_var_names) > 1:
        for i, var_name in enumerate(obs_var_names):
            try:
                y_pp_i = np.asarray(ppc.posterior_predictive[var_name], dtype=float)
            except Exception:
                continue
            panels.append(
                (var_name, y_pp_i, _observed_for_ppc_var(y_obs, obs_rvs, var_name, i))
            )
    elif isinstance(y_obs, dict):
        for feat_name, y_obs_i in y_obs.items():
            y_pp_i = y_pp[feat_name] if isinstance(y_pp, dict) else y_pp
            panels.append((feat_name, np.asarray(y_pp_i, dtype=float), np.asarray(y_obs_i, dtype=float)))
    else:
        panels.append(("", np.asarray(y_pp, dtype=float), y_obs))

    if not panels:
        panels.append(("", np.asarray(y_pp, dtype=float), y_obs))

    n_panels = len(panels)
    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(9, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.5 * n_panels), sharex=True)
        axes = np.atleast_1d(axes)

    for ax, (panel_name, y_pp_panel, y_obs_panel) in zip(axes, panels, strict=True):
        title = "Posterior predictive check"
        if panel_name:
            title = f"{title}: {panel_name}"
        _plot_ppc_flattened_on_ax(ax, y_pp_panel, y_obs_panel, title=title)

    plt.tight_layout()
    plt.show()
    if ppc is not None:
        return ppc
    return {
        "kind": "marginalized_simulated",
        "posterior_predictive": y_pp,
        "observed": y_obs,
    }
