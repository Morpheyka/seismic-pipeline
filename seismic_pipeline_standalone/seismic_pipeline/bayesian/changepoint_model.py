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

from seismic_pipeline.bayesian.priors import (
    _build_mu_regime_normals,
    _build_prior,
    _parse_parameter_selection,
    build_interval_inflated_beta_priors,
)
from seismic_pipeline.features.rem_chunk_features import (
    FIXED_N_CHUNK_DAYS,
    shape_shift_tau_chunk_indices,
)

_SHAPE_SHIFT_METRIC = "shape_shift"


def resolve_n_group_chunks(group_data: dict) -> int:
    """Resolve shared tau axis width from chunk metrics (not shape_shift)."""
    chunk_widths = {
        df.shape[1]
        for feats in group_data.values()
        for metric, df in feats.items()
        if metric != _SHAPE_SHIFT_METRIC
    }
    shape_shift_widths = {
        df.shape[1]
        for feats in group_data.values()
        for metric, df in feats.items()
        if metric == _SHAPE_SHIFT_METRIC
    }
    if chunk_widths:
        if len(chunk_widths) > 1:
            raise ValueError(
                "All chunk metrics must share the same number of columns. "
                f"Widths found: {sorted(chunk_widths)}."
            )
        return max(chunk_widths)
    if shape_shift_widths:
        if len(shape_shift_widths) > 1:
            raise ValueError(
                "All shape_shift blocks must share the same number of columns. "
                f"Widths found: {sorted(shape_shift_widths)}."
            )
        return max(shape_shift_widths)
    raise ValueError("group_data is empty")


def resolve_feat_idx(
    feat_name: str,
    n_cols: int,
    n_group_chunks: int,
    n_days: int = FIXED_N_CHUNK_DAYS,
) -> np.ndarray:
    """Per-column indices for regime switching (maps shape_shift onto chunk tau axis)."""
    if feat_name != _SHAPE_SHIFT_METRIC:
        if n_cols != n_group_chunks:
            raise ValueError(
                f"Feature '{feat_name}' has {n_cols} columns, expected {n_group_chunks}."
            )
        return np.arange(n_group_chunks, dtype=np.int64)
    if n_cols == n_group_chunks:
        return np.arange(n_group_chunks, dtype=np.int64)
    half_n = n_group_chunks // n_days
    expected_shape = n_days - 1
    if n_cols != expected_shape:
        raise ValueError(
            f"shape_shift has {n_cols} columns, expected {expected_shape} (joint) "
            f"or {n_group_chunks} (shape_shift-only)."
        )
    return shape_shift_tau_chunk_indices(n_days, half_n)


def interval_inflated_beta_logp(
    y,
    pi: float,
    alpha: float,
    beta: float,
    threshold: float = 0.9,
):
    """Custom log-probability for Interval-Inflated Beta distribution.

    Parameters
    ----------
    y : tensor, shape (n_obs,)
        Observed values in [0, 1].
    pi : tensor, scalar
        Probability of being in the upper interval [threshold, 1].
    alpha, beta : tensor, scalar
        Beta shape parameters for the lower regime.
    threshold : float
        Cutoff between lower and upper regimes.

    Returns
    -------
    logp : tensor, shape (n_obs,)
    """
    in_upper = pt.ge(y, threshold)
    logp_upper = pt.log(pi) - pt.log(1.0 - threshold)
    logp_lower = pt.log(1.0 - pi) + pm.logp(
        pm.Beta.dist(alpha=alpha, beta=beta), y
    )
    return pt.switch(in_upper, logp_upper, logp_lower)


def sample_interval_inflated_beta(
    rng: np.random.Generator,
    size: int,
    pi: float,
    alpha: float,
    beta: float,
    threshold: float = 0.9,
) -> np.ndarray:
    """Draw from Interval-Inflated Beta: pi * Uniform(threshold, 1) + (1-pi) * Beta(alpha, beta)."""
    pi = float(np.clip(pi, 0.0, 1.0))
    alpha = max(float(alpha), 1e-6)
    beta = max(float(beta), 1e-6)
    threshold = float(threshold)
    upper = rng.uniform(threshold, 1.0, size=size)
    lower = rng.beta(alpha, beta, size=size)
    use_upper = rng.uniform(size=size) < pi
    return np.where(use_upper, upper, lower)


def build_changepoint_model(
    group_data: dict,
    tau_lower: int = 2,
    tau_upper: int | None = None,
    parameter_selection: dict | None = None,
    tau_mode: str = "discrete",
):
    """Build PyMC changepoint model for group_data features.

    Per-metric ``parameter_selection[feat]['g_prior']`` (optional, default ``type: none``)
    scales the **Normal** prior standard deviation on regime means ``mu_*`` by ``sqrt(g)``,
    where ``g`` follows a unit-information, Zellner–Siow, or hyper-``g``/``n`` construction
    (flexible g-prior family from Bayesian model choice literature). Applies to
    ``normal`` / ``student_t`` / ``lognormal`` likelihood blocks (location ``mu``).
    """
    n_group_chunks = resolve_n_group_chunks(group_data)

    if tau_upper is None:
        tau_upper = n_group_chunks
    if not (1 <= tau_lower <= tau_upper - 1 <= n_group_chunks):
        raise ValueError(
            f"Некорректный диапазон tau: [{tau_lower}, {tau_upper}], допустимо [1, {n_group_chunks}]"
        )

    active_features = {feat for features in group_data.values() for feat in features.keys()}
    parameter_cfg = _parse_parameter_selection(parameter_selection, active_features)

    tau_mode = str(tau_mode).strip().lower()
    if tau_mode not in {"discrete", "marginalized"}:
        raise ValueError("Unsupported tau_mode. Use one of: 'discrete', 'marginalized'.")

    with pm.Model() as model:
        idx = np.arange(n_group_chunks)
        tau_values = np.arange(tau_lower, tau_upper + 1, dtype=np.int64)
        n_tau = tau_values.size

        if tau_mode == "discrete":
            tau = pm.DiscreteUniform("tau", lower=tau_lower, upper=tau_upper)
            loglik_by_tau = None
            loglik_by_tau_rows = None
        else:
            loglik_by_tau = np.zeros(n_tau, dtype=float)
            loglik_by_tau_rows = None
            loglik_rows_n: int | None = None

        def _accumulate_marginalized_loglik(
            ll_1,
            ll_2,
            n_rows_obs: int,
            feat_idx: np.ndarray,
        ) -> None:
            nonlocal loglik_by_tau, loglik_by_tau_rows, loglik_rows_n
            feat_mask_before = (feat_idx[None, :] < (tau_values[:, None] - 1)).astype(float)
            feat_mask_before_t = feat_mask_before.T
            feat_mask_after_t = 1.0 - feat_mask_before_t
            ll_tau_rows = pt.dot(ll_1, feat_mask_before_t) + pt.dot(ll_2, feat_mask_after_t)
            ll_tau = ll_tau_rows.sum(axis=0)
            loglik_by_tau = loglik_by_tau + ll_tau
            ll_tau_rows_t = ll_tau_rows.T
            if loglik_by_tau_rows is None:
                loglik_by_tau_rows = ll_tau_rows_t
                loglik_rows_n = int(n_rows_obs)
            else:
                if loglik_rows_n is not None and int(loglik_rows_n) != int(n_rows_obs):
                    raise ValueError(
                        "All selected feature blocks must have the same number of rows for "
                        "marginalized tau WAIC/LOO pointwise aggregation."
                    )
                loglik_by_tau_rows = loglik_by_tau_rows + ll_tau_rows_t

        for group_name, features in group_data.items():
            for feat_name, observed_df in features.items():
                observed = observed_df.to_numpy()
                n_obs_rows = int(observed.shape[0])
                n_cols = int(observed.shape[1])
                feat_idx = resolve_feat_idx(feat_name, n_cols, n_group_chunks)
                spec = parameter_cfg[feat_name]
                likelihood = str(spec.get("likelihood", "normal")).strip().lower()

                if likelihood in {"normal", "student_t", "lognormal"}:
                    mu_1, mu_2 = _build_mu_regime_normals(
                        group_name,
                        feat_name,
                        spec,
                        n_obs_rows=n_obs_rows,
                    )
                    sigma_1 = _build_prior(
                        f"sigma_{group_name}_{feat_name}_1",
                        spec.get("sigma_prior", {"dist": "halfnormal", "sigma": 1.0}),
                        positive_only=True,
                    )
                    sigma_2 = _build_prior(
                        f"sigma_{group_name}_{feat_name}_2",
                        spec.get("sigma_prior", {"dist": "halfnormal", "sigma": 1.0}),
                        positive_only=True,
                    )

                    if likelihood == "normal":
                        if tau_mode == "discrete":
                            mu = pm.math.switch(tau > feat_idx + 1, mu_1, mu_2)
                            sigma = pm.math.switch(tau > feat_idx + 1, sigma_1, sigma_2)
                            pm.Normal(
                                f"obs_{group_name}_{feat_name}",
                                mu=mu,
                                sigma=sigma,
                                observed=observed,
                            )
                        else:
                            ll_1 = pm.logp(pm.Normal.dist(mu=mu_1, sigma=sigma_1), observed)
                            ll_2 = pm.logp(pm.Normal.dist(mu=mu_2, sigma=sigma_2), observed)
                            _accumulate_marginalized_loglik(ll_1, ll_2, n_obs_rows, feat_idx)
                    elif likelihood == "student_t":
                        nu = _build_prior(
                            f"nu_{group_name}_{feat_name}",
                            spec.get("nu_prior", {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0}),
                            positive_only=True,
                        )
                        if tau_mode == "discrete":
                            mu = pm.math.switch(tau > feat_idx + 1, mu_1, mu_2)
                            sigma = pm.math.switch(tau > feat_idx + 1, sigma_1, sigma_2)
                            pm.StudentT(
                                f"obs_{group_name}_{feat_name}",
                                nu=nu,
                                mu=mu,
                                sigma=sigma,
                                observed=observed,
                            )
                        else:
                            ll_1 = pm.logp(pm.StudentT.dist(nu=nu, mu=mu_1, sigma=sigma_1), observed)
                            ll_2 = pm.logp(pm.StudentT.dist(nu=nu, mu=mu_2, sigma=sigma_2), observed)
                            _accumulate_marginalized_loglik(ll_1, ll_2, n_obs_rows, feat_idx)
                    else:
                        if tau_mode == "discrete":
                            mu = pm.math.switch(tau > feat_idx + 1, mu_1, mu_2)
                            sigma = pm.math.switch(tau > feat_idx + 1, sigma_1, sigma_2)
                            pm.LogNormal(
                                f"obs_{group_name}_{feat_name}",
                                mu=mu,
                                sigma=sigma,
                                observed=observed,
                            )
                        else:
                            ll_1 = pm.logp(pm.LogNormal.dist(mu=mu_1, sigma=sigma_1), observed)
                            ll_2 = pm.logp(pm.LogNormal.dist(mu=mu_2, sigma=sigma_2), observed)
                            _accumulate_marginalized_loglik(ll_1, ll_2, n_obs_rows, feat_idx)
                    continue

                if likelihood == "gamma":
                    alpha_1 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_1",
                        spec.get("alpha_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    alpha_2 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_2",
                        spec.get("alpha_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    beta_1 = _build_prior(
                        f"beta_{group_name}_{feat_name}_1",
                        spec.get("beta_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    beta_2 = _build_prior(
                        f"beta_{group_name}_{feat_name}_2",
                        spec.get("beta_prior", {"dist": "exponential", "lam": 1.0}),
                        positive_only=True,
                    )
                    if tau_mode == "discrete":
                        alpha = pm.math.switch(tau > feat_idx + 1, alpha_1, alpha_2)
                        beta = pm.math.switch(tau > feat_idx + 1, beta_1, beta_2)
                        pm.Gamma(
                            f"obs_{group_name}_{feat_name}",
                            alpha=alpha,
                            beta=beta,
                            observed=observed,
                        )
                    else:
                        ll_1 = pm.logp(pm.Gamma.dist(alpha=alpha_1, beta=beta_1), observed)
                        ll_2 = pm.logp(pm.Gamma.dist(alpha=alpha_2, beta=beta_2), observed)
                        _accumulate_marginalized_loglik(ll_1, ll_2, n_obs_rows, feat_idx)
                    continue

                if likelihood == "beta":
                    alpha_1 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_1",
                        spec.get("alpha_prior", {"dist": "exponential_plus", "lam": 0.5, "offset": 1.0}),
                        positive_only=True,
                    )
                    alpha_2 = _build_prior(
                        f"alpha_{group_name}_{feat_name}_2",
                        spec.get("alpha_prior", {"dist": "exponential_plus", "lam": 0.5, "offset": 1.0}),
                        positive_only=True,
                    )
                    beta_1 = _build_prior(
                        f"beta_{group_name}_{feat_name}_1",
                        spec.get("beta_prior", {"dist": "exponential_plus", "lam": 0.5, "offset": 1.0}),
                        positive_only=True,
                    )
                    beta_2 = _build_prior(
                        f"beta_{group_name}_{feat_name}_2",
                        spec.get("beta_prior", {"dist": "exponential_plus", "lam": 0.5, "offset": 1.0}),
                        positive_only=True,
                    )
                    # Beta support is (0, 1), so clip normalized data slightly
                    # away from boundaries to avoid -inf logp at exact 0/1.
                    beta_eps = float(spec.get("eps", 1e-6))
                    observed_beta = np.clip(observed, beta_eps, 1.0 - beta_eps)
                    if tau_mode == "discrete":
                        alpha = pm.math.switch(tau > feat_idx + 1, alpha_1, alpha_2)
                        beta = pm.math.switch(tau > feat_idx + 1, beta_1, beta_2)
                        pm.Beta(
                            f"obs_{group_name}_{feat_name}",
                            alpha=alpha,
                            beta=beta,
                            observed=observed_beta,
                        )
                    else:
                        ll_1 = pm.logp(pm.Beta.dist(alpha=alpha_1, beta=beta_1), observed_beta)
                        ll_2 = pm.logp(pm.Beta.dist(alpha=alpha_2, beta=beta_2), observed_beta)
                        _accumulate_marginalized_loglik(ll_1, ll_2, n_obs_rows, feat_idx)
                    continue

                if likelihood == "interval_inflated_beta":
                    threshold = float(spec.get("threshold", 0.9))
                    priors_1 = build_interval_inflated_beta_priors(
                        feat_name=f"{group_name}_{feat_name}",
                        regime=1,
                        threshold=threshold,
                        pi_prior=spec.get("pi_prior"),
                        alpha_prior=spec.get("alpha_prior"),
                        beta_prior=spec.get("beta_prior"),
                    )
                    priors_2 = build_interval_inflated_beta_priors(
                        feat_name=f"{group_name}_{feat_name}",
                        regime=2,
                        threshold=threshold,
                        pi_prior=spec.get("pi_prior"),
                        alpha_prior=spec.get("alpha_prior"),
                        beta_prior=spec.get("beta_prior"),
                    )
                    pi_1, alpha_1, beta_1 = priors_1["pi"], priors_1["alpha"], priors_1["beta"]
                    pi_2, alpha_2, beta_2 = priors_2["pi"], priors_2["alpha"], priors_2["beta"]
                    thr = priors_1["threshold"]

                    eps = float(spec.get("eps", 1e-6))
                    observed_clipped = np.clip(observed, eps, 1.0 - eps)
                    y_obs = pt.as_tensor_variable(observed_clipped)

                    if tau_mode == "discrete":
                        logp_1 = interval_inflated_beta_logp(y_obs, pi_1, alpha_1, beta_1, thr)
                        logp_2 = interval_inflated_beta_logp(y_obs, pi_2, alpha_2, beta_2, thr)
                        regime_before = pt.cast(tau > feat_idx + 1, "float64")[None, :]
                        pm.Potential(
                            f"obs_{group_name}_{feat_name}",
                            pt.sum(regime_before * logp_1 + (1.0 - regime_before) * logp_2),
                        )
                    else:
                        ll_1 = interval_inflated_beta_logp(y_obs, pi_1, alpha_1, beta_1, thr)
                        ll_2 = interval_inflated_beta_logp(y_obs, pi_2, alpha_2, beta_2, thr)
                        _accumulate_marginalized_loglik(ll_1, ll_2, n_obs_rows, feat_idx)
                    continue

                raise ValueError(
                    f"Unsupported likelihood '{likelihood}' for feature '{feat_name}'. "
                    "Use one of: normal, student_t, lognormal, gamma, beta, interval_inflated_beta."
                )

        if tau_mode == "marginalized":
            # p(y|theta) = logsumexp_k [ log p(y|tau=k, theta) + log p(tau=k) ]
            log_w = loglik_by_tau - np.log(float(n_tau))
            log_w_rows = loglik_by_tau_rows - np.log(float(n_tau))
            # Pointwise (per-row) marginalized log-likelihood for WAIC/LOO.
            pointwise_log_lik = pm.math.logsumexp(log_w_rows, axis=0)
            pm.Deterministic("changepoint_pointwise_log_lik", pointwise_log_lik)
            # Scalar joint log-likelihood (kept for diagnostics / backward compatibility).
            pm.Deterministic("changepoint_joint_log_lik", pm.math.logsumexp(log_w))
            pm.Potential("tau_marginalized_logp", pm.math.logsumexp(log_w))
            tau_probs = pm.Deterministic("tau_probs", pm.math.softmax(log_w))
            tau_support = pm.Deterministic(
                "tau_support",
                pt.as_tensor_variable(tau_values.astype(np.float64)),
            )
            pm.Deterministic("tau_mean", pm.math.sum(tau_probs * tau_support))

    return model
def sample_model(
    model,
    draws: int = 4000,
    tune: int = 2000,
    *,
    nuts_backend: str = "pymc",
    chains: int = 4,
    cores: int | None = None,
    progressbar: bool = True,
):
    """Run MCMC sampling and return MultiTrace.

    Parameters
    ----------
    nuts_backend:
        - "pymc" (default): classic PyMC NUTS
        - "numpyro": JAX/NumPyro NUTS backend (can use GPU)
        - "blackjax": JAX/BlackJAX NUTS backend (can use GPU)
    """
    backend = str(nuts_backend).lower().strip()
    sample_kwargs = dict(
        draws=draws,
        tune=tune,
        return_inferencedata=False,
        compute_convergence_checks=False,
        target_accept=0.9,
        chains=chains,
        progressbar=bool(progressbar),
    )
    if cores is not None:
        sample_kwargs["cores"] = cores

    with model:
        if backend == "pymc":
            trace = pm.sample(**sample_kwargs, init="jitter+adapt_diag")
        elif backend in {"numpyro", "blackjax"}:
            import jax

            # With a single device, JAX "parallel" chains via pmap may fail for chains>1.
            device_count = int(jax.device_count())
            nuts_sampler_kwargs: dict[str, object] = {}
            jax_vectorized = chains > 1 and device_count < chains
            if jax_vectorized:
                nuts_sampler_kwargs["chain_method"] = "vectorized"
            # BlackJAX progress bar uses IO callbacks that can fail under vectorized chains.
            if jax_vectorized:
                sample_kwargs["progressbar"] = False

            try:
                trace = pm.sample(
                    nuts_sampler=backend,
                    nuts_sampler_kwargs=nuts_sampler_kwargs,
                    **sample_kwargs,
                )
            except ValueError as exc:
                if "Model can not be sampled with NUTS alone" in str(exc):
                    raise ValueError(
                        "JAX backend requires a fully continuous differentiable model. "
                        "Use tau_mode='marginalized' (or backend='pymc')."
                    ) from exc
                raise
        else:
            raise ValueError(
                "Unsupported nuts_backend. Use one of: 'pymc', 'numpyro', 'blackjax'."
            )
    return trace
