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

def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out
def _default_parameter_selection() -> dict[str, dict]:
    """Per-metric defaults; optional ``g_prior`` scales Normal priors on regime means (Zellner-style).

    ``g_prior`` keys:
    - ``type``: ``none`` | ``unit_information`` | ``hyper_g_n`` | ``zellner_siow``
    - ``n``: effective sample size (default: number of rows in observed matrix for that block)
    - ``a``, ``b``: Beta hyperparameters for ``hyper_g_n`` (default 3, 3 as in Liang et al. style demos)
    """
    g_none = {"type": "none"}
    return {
        "mean": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 1.5},
            "sigma_prior": {"dist": "halfnormal", "sigma": 1.0},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
            "g_prior": dict(g_none),
        },
        "range": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -0.3, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.7},
            "g_prior": dict(g_none),
        },
        "std": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -0.7, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.7},
            "g_prior": dict(g_none),
        },
        "skewness": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 2.5},
            "sigma_prior": {"dist": "halfstudentt", "nu": 4.0, "sigma": 1.5},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
            "g_prior": dict(g_none),
        },
        "kurtosis": {
            "likelihood": "student_t",
            "mu_prior": {"dist": "normal", "mu": 0.0, "sigma": 3.0},
            "sigma_prior": {"dist": "halfstudentt", "nu": 4.0, "sigma": 2.0},
            "nu_prior": {"dist": "exponential_plus", "lam": 0.05, "offset": 2.0},
            "g_prior": dict(g_none),
        },
        "shape_shift": {
            "likelihood": "lognormal",
            "mu_prior": {"dist": "normal", "mu": -2.0, "sigma": 1.0},
            "sigma_prior": {"dist": "halfnormal", "sigma": 0.5},
            "g_prior": dict(g_none),
        },
    }
def parameter_selection_with_g_prior(
    base: dict[str, dict] | None,
    g_type: str,
    *,
    n: int | None = None,
    a: float = 3.0,
    b: float = 3.0,
) -> dict[str, dict]:
    """Return a copy of ``base`` (or defaults) with ``g_prior`` set for every metric key present."""
    defaults = _default_parameter_selection()
    src = dict(base) if base else {}
    out: dict[str, dict] = {}
    for feat_name in defaults:
        merged = _deep_merge_dict(defaults[feat_name], src.get(feat_name, {}) or {})
        gp = dict(merged.get("g_prior") or {})
        gp["type"] = str(g_type).strip().lower()
        if n is not None:
            gp["n"] = int(n)
        gp.setdefault("a", float(a))
        gp.setdefault("b", float(b))
        merged["g_prior"] = gp
        out[feat_name] = merged
    return out

VALID_LIKELIHOODS: dict[str, list[str]] = {
    "mean": ["student_t", "lognormal", "normal", "skew_normal"],
    "range": [
        "beta",
        "beta_constrained",
        "normal",
        "lognormal",
        "interval_inflated_beta",
        "zero_inflated_beta",
    ],
    "std": ["student_t", "lognormal", "gamma"],
    "skewness": ["student_t"],
    "kurtosis": ["student_t"],
    "shape_shift": ["lognormal", "gamma"],
}

# Default shape priors for plain Beta allow α,β < 1 (U-shape / boundary spike).
# Constrained Beta forces α,β ≥ 1 via Gamma_offset (raw Gamma + 1).
CONSTRAINED_BETA_SHAPE_PRIOR: dict[str, float | str] = {
    "dist": "gamma_offset",
    "mu": 2.0,
    "sigma": 1.0,
    "offset": 1.0,
}


def constrained_beta_shape_prior() -> dict[str, float | str]:
    """Prior for Beta shape params with support ≥ 1 (no U-shape / boundary spike)."""
    return dict(CONSTRAINED_BETA_SHAPE_PRIOR)


def _parse_parameter_selection(parameter_selection, active_features: set[str]) -> dict[str, dict]:
    """Return fully expanded per-feature distribution config."""
    defaults = _default_parameter_selection()
    if parameter_selection is None:
        parameter_selection = {}
    if not isinstance(parameter_selection, dict):
        raise ValueError("parameter_selection must be dict keyed by feature name.")

    unknown = sorted(set(parameter_selection.keys()) - set(defaults.keys()))
    if unknown:
        raise ValueError(
            f"Unknown feature keys in parameter_selection: {unknown}. "
            f"Use one of: {sorted(defaults.keys())}"
        )

    out: dict[str, dict] = {}
    for feature_name in sorted(active_features):
        base_cfg = defaults[feature_name]
        custom_cfg = parameter_selection.get(feature_name, {})
        if custom_cfg is None:
            custom_cfg = {}
        if not isinstance(custom_cfg, dict):
            raise ValueError(f"parameter_selection['{feature_name}'] must be a dict.")
        merged = _deep_merge_dict(base_cfg, custom_cfg)
        likelihood = str(merged.get("likelihood", "")).strip().lower()
        allowed = VALID_LIKELIHOODS.get(feature_name, [])
        if likelihood not in allowed:
            raise ValueError(
                f"Unsupported likelihood '{likelihood}' for feature '{feature_name}'. "
                f"Use one of: {allowed}"
            )
        out[feature_name] = merged
    return out
def _build_prior(var_name: str, spec: dict, *, positive_only: bool = False):
    if not isinstance(spec, dict):
        raise ValueError(f"Prior spec for '{var_name}' must be a dict.")
    dist = str(spec.get("dist", "")).strip().lower()
    if not dist:
        raise ValueError(f"Prior spec for '{var_name}' must include 'dist'.")

    positive_dists = {
        "halfnormal",
        "halfstudentt",
        "exponential",
        "lognormal",
        "exponential_plus",
        "gamma",
        "gamma_offset",
        "gamma_plus",
        "truncated_gamma",
    }
    if positive_only and dist not in positive_dists:
        raise ValueError(
            f"Prior '{var_name}' must be positive; use one of {sorted(positive_dists)}, got '{dist}'."
        )

    if dist == "normal":
        return pm.Normal(var_name, mu=float(spec.get("mu", 0.0)), sigma=float(spec.get("sigma", 1.0)))
    if dist == "halfnormal":
        return pm.HalfNormal(var_name, sigma=float(spec.get("sigma", 1.0)))
    if dist == "halfstudentt":
        return pm.HalfStudentT(
            var_name,
            nu=float(spec.get("nu", 4.0)),
            sigma=float(spec.get("sigma", 1.0)),
        )
    if dist == "exponential":
        return pm.Exponential(var_name, lam=float(spec.get("lam", 1.0)))
    if dist == "lognormal":
        return pm.LogNormal(
            var_name,
            mu=float(spec.get("mu", 0.0)),
            sigma=float(spec.get("sigma", 1.0)),
        )
    if dist == "exponential_plus":
        lam = float(spec.get("lam", 0.05))
        offset = float(spec.get("offset", 2.0))
        raw = pm.Exponential(f"{var_name}_raw", lam=lam)
        return pm.Deterministic(var_name, raw + offset)
    if dist == "beta":
        return pm.Beta(
            var_name,
            alpha=float(spec.get("alpha", 1.0)),
            beta=float(spec.get("beta", 1.0)),
        )
    if dist == "gamma":
        if "alpha" in spec and "beta" in spec:
            return pm.Gamma(
                var_name,
                alpha=float(spec["alpha"]),
                beta=float(spec["beta"]),
            )
        mu = float(spec.get("mu", 1.0))
        sigma = float(spec.get("sigma", 1.0))
        if sigma <= 0.0:
            raise ValueError(f"gamma prior sigma must be > 0 for '{var_name}'.")
        return pm.Gamma(var_name, mu=mu, sigma=sigma)

    if dist in {"gamma_offset", "gamma_plus"}:
        # Gamma(mu, sigma) + offset → support [offset, ∞). Used for β-shape ≥ 1.
        offset = float(spec.get("offset", 1.0))
        raw_spec = {k: v for k, v in spec.items() if k not in {"dist", "offset"}}
        raw_spec["dist"] = "gamma"
        raw = _build_prior(f"{var_name}_raw", raw_spec, positive_only=True)
        return pm.Deterministic(var_name, raw + offset)

    if dist == "truncated_gamma":
        lower = float(spec.get("lower", 1.0))
        upper = spec.get("upper", None)
        if "alpha" in spec and "beta" in spec:
            base = pm.Gamma.dist(alpha=float(spec["alpha"]), beta=float(spec["beta"]))
        else:
            mu = float(spec.get("mu", 3.0))
            sigma = float(spec.get("sigma", 1.5))
            if sigma <= 0.0:
                raise ValueError(f"truncated_gamma prior sigma must be > 0 for '{var_name}'.")
            base = pm.Gamma.dist(mu=mu, sigma=sigma)
        kwargs: dict[str, Any] = {"lower": lower}
        if upper is not None:
            kwargs["upper"] = float(upper)
        return pm.Truncated(var_name, base, **kwargs)

    raise ValueError(
        f"Unsupported prior dist '{dist}' for '{var_name}'. "
        "Use one of: normal, halfnormal, halfstudentt, exponential, lognormal, "
        "exponential_plus, beta, gamma, gamma_offset, truncated_gamma."
    )


def build_interval_inflated_beta_priors(
    feat_name: str,
    regime: int,
    threshold: float = 0.9,
    pi_prior: dict | None = None,
    alpha_prior: dict | None = None,
    beta_prior: dict | None = None,
) -> dict:
    """Build prior random variables for Interval-Inflated Beta.

    Parameters
    ----------
    feat_name : str
        Feature name prefix (e.g., ``concat_range``).
    regime : int
        Regime index (1 = before tau, 2 = after tau).
    threshold : float
        Cutoff between lower and upper regimes.
    pi_prior : dict, optional
        Prior spec for pi. Default: Beta(1, 10) — expects rare saturation.
    alpha_prior : dict, optional
        Prior spec for alpha. Default: Gamma(mu=3, sigma=1).
    beta_prior : dict, optional
        Prior spec for beta. Default: Gamma(mu=3, sigma=1).

    Returns
    -------
    dict with keys: ``pi``, ``alpha``, ``beta``, ``threshold``, ``likelihood_type``
    """
    if pi_prior is None:
        pi_prior = {"dist": "beta", "alpha": 1.0, "beta": 10.0}
    if alpha_prior is None:
        alpha_prior = {"dist": "gamma", "mu": 3.0, "sigma": 1.0}
    if beta_prior is None:
        beta_prior = {"dist": "gamma", "mu": 3.0, "sigma": 1.0}

    name_prefix = f"{feat_name}_{regime}"

    pi = _build_prior(
        f"pi_{name_prefix}",
        pi_prior,
        positive_only=False,
    )
    alpha = _build_prior(
        f"alpha_{name_prefix}",
        alpha_prior,
        positive_only=True,
    )
    beta = _build_prior(
        f"beta_{name_prefix}",
        beta_prior,
        positive_only=True,
    )

    return {
        "pi": pi,
        "alpha": alpha,
        "beta": beta,
        "threshold": float(threshold),
        "likelihood_type": "interval_inflated_beta",
    }


def build_zero_inflated_beta_priors(
    feat_name: str,
    regime: int,
    pi_prior: dict | None = None,
    alpha_prior: dict | None = None,
    beta_prior: dict | None = None,
) -> dict:
    """Build prior RVs for Zero-Inflated Beta (point mass at 0 + Beta on (0, 1)).

    ``pi`` = P(y ≈ 0). Default Beta(1, 10) expects rare exact zeros.
    """
    if pi_prior is None:
        pi_prior = {"dist": "beta", "alpha": 1.0, "beta": 10.0}
    if alpha_prior is None:
        alpha_prior = {"dist": "gamma", "mu": 3.0, "sigma": 1.0}
    if beta_prior is None:
        beta_prior = {"dist": "gamma", "mu": 3.0, "sigma": 1.0}

    name_prefix = f"{feat_name}_{regime}"
    pi = _build_prior(f"pi_{name_prefix}", pi_prior, positive_only=False)
    alpha = _build_prior(f"alpha_{name_prefix}", alpha_prior, positive_only=True)
    beta = _build_prior(f"beta_{name_prefix}", beta_prior, positive_only=True)
    return {
        "pi": pi,
        "alpha": alpha,
        "beta": beta,
        "likelihood_type": "zero_inflated_beta",
    }
def _g_multiplier(name_prefix: str, n_obs_rows: int, g_prior: dict | None):
    """Scalar g for Zellner-style scaling of Normal mu priors (Chapter 8 style flexible g).

    - ``none``: g = 1 (standard fixed prior spread).
    - ``unit_information``: g = n (fixed).
    - ``zellner_siow``: n/g ~ Gamma(1/2, 1/2)  =>  g = n / (n/g).
    - ``hyper_g_n``: u = 1/(1+n/g) ~ Beta(a/2, b/2)  =>  g = n*u/(1-u).

    Prior std on each regime mean uses ``mu_prior['sigma'] * sqrt(g)``.
    """
    if not g_prior:
        g_prior = {}
    typ = str(g_prior.get("type", "none")).strip().lower()
    if typ in {"", "none", "off", "fixed"}:
        return pt.as_tensor_variable(np.asarray(1.0, dtype=np.float64))

    n = int(g_prior.get("n", n_obs_rows))
    n = max(n, 1)
    n_f = float(n)

    if typ == "unit_information":
        return pt.as_tensor_variable(np.asarray(n_f, dtype=np.float64))

    if typ == "zellner_siow":
        n_over_g = pm.Gamma(f"{name_prefix}_n_over_g", alpha=0.5, beta=0.5)
        return pm.Deterministic(f"{name_prefix}_g", n_f / (n_over_g + 1e-12))

    if typ == "hyper_g_n":
        a = float(g_prior.get("a", 3.0))
        b = float(g_prior.get("b", 3.0))
        u = pm.Beta(f"{name_prefix}_u_hyper_gn", alpha=a * 0.5, beta=b * 0.5)
        one_m_u = pt.clip(1.0 - u, 1e-6, 1.0)
        return pm.Deterministic(f"{name_prefix}_g", n_f * u / one_m_u)

    raise ValueError(
        f"Unknown g_prior type '{typ}' for '{name_prefix}'. "
        "Use one of: none, unit_information, hyper_g_n, zellner_siow."
    )
def _build_mu_regime_normals(
    group_name: str,
    feat_name: str,
    spec: dict,
    *,
    n_obs_rows: int,
) -> tuple[object, object]:
    """Normal priors on mu_1, mu_2 for location parameter (Normal / StudentT / LogNormal mu)."""
    mu_spec = spec.get("mu_prior", {"dist": "normal", "mu": 0.0, "sigma": 2.0})
    if not isinstance(mu_spec, dict):
        raise ValueError(f"mu_prior for {group_name}/{feat_name} must be a dict.")
    b0_1 = float(mu_spec.get("mu", 0.0))
    b0_2 = float(mu_spec.get("mu_2", mu_spec.get("mu", 0.0)))
    s0 = float(mu_spec.get("sigma", 1.0))
    if s0 <= 0.0:
        raise ValueError(f"mu_prior.sigma must be > 0 for {group_name}/{feat_name}.")

    prefix = f"g_{group_name}_{feat_name}"
    g_prior = spec.get("g_prior")
    g = _g_multiplier(prefix, n_obs_rows, g_prior if isinstance(g_prior, dict) else None)
    scale = s0 * pt.sqrt(g)

    mu_1 = pm.Normal(f"mu_{group_name}_{feat_name}_1", mu=b0_1, sigma=scale)
    mu_2 = pm.Normal(f"mu_{group_name}_{feat_name}_2", mu=b0_2, sigma=scale)
    return mu_1, mu_2
