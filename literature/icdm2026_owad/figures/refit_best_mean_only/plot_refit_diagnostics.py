#!/usr/bin/env python3
"""Diagnostic figures for a single changepoint refit (mean / range, any likelihood).

Reads ``trace.nc`` or ``trace.zarr``, ``refit_meta.json``, and ``observations.npz``
from a ``run_top10_refit_plots.py`` output directory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.special import gammaln
from scipy.stats import gaussian_kde, skewnorm

REPO = Path(__file__).resolve().parents[4]
STANDALONE = REPO / "seismic_pipeline_standalone"
sys.path.insert(0, str(STANDALONE))

from seismic_pipeline.bayesian.diagnostics import (  # noqa: E402
    _likelihood_pdf_from_posterior,
    _observed_split_by_tau,
    _profile_x_grid,
    _tau_map_from_trace,
    _values_by_chain,
    _values_flat,
    interval_inflated_beta_pdf_mixture,
    tau_probabilities,
)

COLOR_BEFORE = "#A60628"
COLOR_AFTER = "#7A68A6"
COLOR_TAU = "#348ABD"
COLOR_EXTRA = "#4E79A7"


def setup_style(base: int = 11) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
            "font.size": base,
            "axes.titlesize": base + 1,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=450)
    fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)
    print(f"wrote {out_dir / stem}.png")


def load_trace(refit_dir: Path):
    nc = refit_dir / "trace.nc"
    zarr = refit_dir / "trace.zarr"
    if nc.is_file():
        return xr.open_datatree(str(nc))
    if zarr.exists():
        return xr.open_datatree(str(zarr), engine="zarr")
    raise FileNotFoundError(f"Neither trace.nc nor trace.zarr under {refit_dir}")


def load_meta(refit_dir: Path) -> dict:
    return json.loads((refit_dir / "refit_meta.json").read_text(encoding="utf-8"))


def load_observations(refit_dir: Path) -> dict[str, np.ndarray]:
    path = refit_dir / "observations.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {k: np.asarray(z[k], dtype=float) for k in z.files}


def parse_likelihood(meta: dict) -> tuple[str, str]:
    """Return (feature_name, likelihood) from meta."""
    lik_raw = str(meta.get("likelihoods", ""))
    cfg = meta.get("config") or {}
    param_sel = cfg.get("parameter_selection") or {}
    # Prefer config
    for feat, spec in param_sel.items():
        if isinstance(spec, dict) and "likelihood" in spec:
            return str(feat), str(spec["likelihood"]).strip().lower()
    # Fallback parse "mean=skew_normal" / "range=interval_inflated_beta"
    m = re.search(r"(mean|range)=([a-z0-9_]+)", lik_raw)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse likelihood from meta: {lik_raw!r}")


def discover_obs_key(obs: dict[str, np.ndarray], feature: str) -> str:
    preferred = [
        f"daily__{feature}",
        f"concat__{feature}",
        f"day__{feature}",
        f"night__{feature}",
    ]
    for key in preferred:
        if key in obs:
            return key
    for key in sorted(obs):
        if key.endswith(f"__{feature}"):
            return key
    raise ValueError(f"No {feature!r} observations in {list(obs)}")


def discover_param_pairs(trace, feature: str) -> dict[str, tuple[str, str] | str | None]:
    """Find before/after parameter name pairs for the active feature."""
    vars_ = set(trace.posterior.data_vars)
    # Prefer daily_* then concat_*
    group_candidates = ["daily", "concat", "day", "night"]

    def pair(prefix: str) -> tuple[str, str] | None:
        for g in group_candidates:
            p1 = f"{prefix}_{g}_{feature}_1"
            p2 = f"{prefix}_{g}_{feature}_2"
            if p1 in vars_ and p2 in vars_:
                return p1, p2
        # any match
        for v in sorted(vars_):
            if v.startswith(f"{prefix}_") and v.endswith(f"_{feature}_1"):
                p2 = v[:-1] + "2"
                if p2 in vars_:
                    return v, p2
        return None

    def shared(prefix: str) -> str | None:
        for g in group_candidates:
            name = f"{prefix}_{g}_{feature}"
            if name in vars_:
                return name
        for v in sorted(vars_):
            if v.startswith(f"{prefix}_") and v.endswith(f"_{feature}") and not v.endswith(("_1", "_2")):
                return v
        return None

    nu_pair = pair("nu")
    return {
        "mu": pair("mu"),
        "sigma": pair("sigma"),
        "alpha": pair("alpha"),
        "beta": pair("beta"),
        "pi": pair("pi"),
        "nu": nu_pair if nu_pair is not None else shared("nu"),
    }


def active_trace_vars(pairs: dict) -> list[str]:
    out: list[str] = []
    for key, val in pairs.items():
        if isinstance(val, tuple):
            out.extend(list(val))
        elif isinstance(val, str):
            out.append(val)
    return out


def plot_traces_compact(trace, var_names: list[str], out_dir: Path, title: str) -> None:
    setup_style(10)
    names = [v for v in var_names if v in set(trace.posterior.data_vars)]
    if "tau_mean" in set(trace.posterior.data_vars):
        names.append("tau_mean")
    if not names:
        raise ValueError("No trace variables to plot")

    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 1.85 * n), sharex="col")
    if n == 1:
        axes = np.asarray(axes).reshape(1, 2)

    chain_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    for i, var in enumerate(names):
        chains = _values_by_chain(trace, var)
        ax_tr, ax_den = axes[i, 0], axes[i, 1]
        for c_idx, chain in enumerate(chains):
            color = chain_colors[c_idx % len(chain_colors)]
            ax_tr.plot(chain, color=color, alpha=0.85, lw=0.6)
            vals = np.asarray(chain, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size >= 5 and np.unique(vals).size >= 3:
                try:
                    kde = gaussian_kde(vals)
                    lo, hi = np.percentile(vals, [1, 99])
                    xs = np.linspace(lo, hi, 250)
                    ax_den.plot(xs, kde(xs), color=color, lw=1.3, alpha=0.9)
                except Exception:
                    ax_den.hist(vals, bins=40, density=True, color=color, alpha=0.35)
            else:
                ax_den.hist(vals, bins=40, density=True, color=color, alpha=0.35)
        ax_tr.set_ylabel(var.replace("_", "\n"), fontsize=8)
        if i == n - 1:
            ax_tr.set_xlabel("draw")
            ax_den.set_xlabel("value")
        if i == 0:
            ax_tr.set_title("trace", loc="left", fontweight="bold")
            ax_den.set_title("density", loc="left", fontweight="bold")

    fig.suptitle(f"MCMC traces — {title}", fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "01_traces")


def _kde_or_hist(ax, values: np.ndarray, *, color: str, label: str) -> float:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    mean_v = float(np.mean(values))
    if values.size < 5 or np.unique(values).size < 3:
        ax.hist(values, bins=30, density=True, alpha=0.55, color=color, label=label, edgecolor="white")
        return mean_v
    try:
        kde = gaussian_kde(values)
        lo, hi = np.percentile(values, [0.5, 99.5])
        xs = np.linspace(lo, hi, 400)
        ax.fill_between(xs, kde(xs), color=color, alpha=0.35, label=label)
        ax.plot(xs, kde(xs), color=color, lw=1.8)
    except Exception:
        ax.hist(values, bins=30, density=True, alpha=0.55, color=color, label=label, edgecolor="white")
    return mean_v


def plot_params_overlay(trace, pairs: dict, out_dir: Path, title: str) -> None:
    setup_style(11)
    panels: list[tuple[str, str, object]] = []
    # Prefer informative order by likelihood family
    for key, label in (
        ("mu", r"$\mu$"),
        ("sigma", r"$\sigma$"),
        ("alpha", r"$\alpha$"),
        ("beta", r"$\beta$"),
        ("pi", r"$\pi$"),
    ):
        if pairs.get(key):
            panels.append((key, label, pairs[key]))
    if pairs.get("nu"):
        panels.append(("nu", r"$\nu$", pairs["nu"]))

    # Layout: param panels + tau
    n_param = len(panels)
    n_total = n_param + 1
    n_cols = 2
    n_rows = int(np.ceil(n_total / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for i, (key, label, val) in enumerate(panels):
        ax = axes[i]
        if isinstance(val, str):
            m = _kde_or_hist(ax, _values_flat(trace, val), color=COLOR_EXTRA, label=label)
            ax.axvline(m, color="#222222", ls="--", lw=1.2)
        else:
            p1, p2 = val  # type: ignore[misc]
            m1 = _kde_or_hist(ax, _values_flat(trace, p1), color=COLOR_BEFORE, label=r"before $\tau$")
            m2 = _kde_or_hist(ax, _values_flat(trace, p2), color=COLOR_AFTER, label=r"after $\tau$")
            ax.axvline(m1, color=COLOR_BEFORE, ls="--", lw=1.2, alpha=0.9)
            ax.axvline(m2, color=COLOR_AFTER, ls="--", lw=1.2, alpha=0.9)
        letter = chr(ord("a") + i)
        ax.set_title(f"({letter}) {label}", loc="left", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(loc="best")

    # tau panel
    ax = axes[n_param]
    support, probs = tau_probabilities(trace)
    support = np.asarray(support, dtype=int)
    probs = np.asarray(probs, dtype=float)
    e_tau = float(np.sum(support * probs))
    ax.bar(support, probs, color=COLOR_TAU, width=0.75, edgecolor="white", linewidth=0.6)
    ax.axvline(e_tau, color="#E15759", ls="--", lw=1.6, label=rf"$\mathbb{{E}}[\tau]={e_tau:.2f}$")
    ax.set_xticks(support)
    ax.set_xlabel(r"$\tau$ (day index)")
    ax.set_ylabel(r"$P(\tau=k)$")
    letter = chr(ord("a") + n_param)
    ax.set_title(rf"({letter}) onset $\tau$", loc="left", fontweight="bold")
    ax.legend(loc="best")

    for j in range(n_param + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Parameter posteriors — {title}", fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "02_params_overlay")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(support, probs, color=COLOR_TAU, width=0.75, edgecolor="white", linewidth=0.6)
    ax.axvline(e_tau, color="#E15759", ls="--", lw=1.6, label=rf"$\mathbb{{E}}[\tau]={e_tau:.2f}$")
    ax.set_xticks(support)
    ax.set_xlabel(r"$\tau$ (day index)")
    ax.set_ylabel(r"$P(\tau=k)$")
    ax.set_title(f"Onset posterior — {title}", loc="left", fontweight="bold")
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, out_dir, "03_tau_posterior")


def _mean_params(trace, pairs: dict, likelihood: str, threshold: float = 0.9) -> tuple[dict, dict]:
    p1: dict[str, float] = {}
    p2: dict[str, float] = {}
    for key in ("mu", "sigma", "alpha", "beta", "pi", "nu"):
        val = pairs.get(key)
        if isinstance(val, tuple):
            p1[key] = float(np.mean(_values_flat(trace, val[0])))
            p2[key] = float(np.mean(_values_flat(trace, val[1])))
        elif isinstance(val, str) and key == "nu":
            nu = float(np.mean(_values_flat(trace, val)))
            p1["nu"] = nu
            p2["nu"] = nu
    if likelihood == "interval_inflated_beta":
        p1["threshold"] = threshold
        p2["threshold"] = threshold
    return p1, p2


def _pdf_batch(likelihood: str, x: np.ndarray, draws: dict[str, np.ndarray], threshold: float = 0.9) -> np.ndarray:
    """Shape (n_draws, n_x)."""
    n = len(next(iter(draws.values())))
    x = np.asarray(x, dtype=float)
    out = np.zeros((n, x.size), dtype=float)
    if likelihood == "student_t":
        mu = draws["mu"][:, None]
        sigma = np.maximum(draws["sigma"][:, None], 1e-12)
        nu = np.maximum(draws["nu"][:, None], 2.0 + 1e-12)
        xv = x[None, :]
        log_c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi) - np.log(sigma)
        z2 = ((xv - mu) / sigma) ** 2
        return np.exp(log_c - ((nu + 1.0) / 2.0) * np.log1p(z2 / nu))
    if likelihood == "skew_normal":
        for i in range(n):
            out[i] = skewnorm.pdf(x, a=float(draws["alpha"][i]), loc=float(draws["mu"][i]), scale=max(float(draws["sigma"][i]), 1e-12))
        return out
    if likelihood == "interval_inflated_beta":
        for i in range(n):
            out[i] = interval_inflated_beta_pdf_mixture(
                x,
                float(draws["pi"][i]),
                max(float(draws["alpha"][i]), 1e-12),
                max(float(draws["beta"][i]), 1e-12),
                threshold,
            )
        return out
    raise ValueError(f"Unsupported likelihood for fan: {likelihood}")


def _draw_arrays(trace, pairs: dict, likelihood: str, idx: np.ndarray) -> dict[str, np.ndarray]:
    keys = {
        "student_t": ("mu", "sigma", "nu"),
        "skew_normal": ("mu", "sigma", "alpha"),
        "interval_inflated_beta": ("alpha", "beta", "pi"),
    }[likelihood]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key == "nu":
            out[key] = _values_flat(trace, pairs["nu"])[idx]  # type: ignore[index]
        else:
            # for fan we pass one regime at a time
            raise AssertionError("use _draw_arrays_regime")
    return out


def _draw_arrays_regime(trace, pairs: dict, likelihood: str, regime: int, idx: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if likelihood == "student_t":
        out["mu"] = _values_flat(trace, pairs["mu"][regime - 1])[idx]  # type: ignore[index]
        out["sigma"] = _values_flat(trace, pairs["sigma"][regime - 1])[idx]  # type: ignore[index]
        nu_val = pairs["nu"]
        if isinstance(nu_val, tuple):
            out["nu"] = _values_flat(trace, nu_val[regime - 1])[idx]
        else:
            out["nu"] = _values_flat(trace, nu_val)[idx]  # type: ignore[arg-type]
    elif likelihood == "skew_normal":
        out["mu"] = _values_flat(trace, pairs["mu"][regime - 1])[idx]  # type: ignore[index]
        out["sigma"] = _values_flat(trace, pairs["sigma"][regime - 1])[idx]  # type: ignore[index]
        out["alpha"] = _values_flat(trace, pairs["alpha"][regime - 1])[idx]  # type: ignore[index]
    elif likelihood == "interval_inflated_beta":
        out["alpha"] = _values_flat(trace, pairs["alpha"][regime - 1])[idx]  # type: ignore[index]
        out["beta"] = _values_flat(trace, pairs["beta"][regime - 1])[idx]  # type: ignore[index]
        out["pi"] = _values_flat(trace, pairs["pi"][regime - 1])[idx]  # type: ignore[index]
    else:
        raise ValueError(likelihood)
    return out


def _mean_pdf(likelihood: str, x: np.ndarray, p1: dict, p2: dict) -> tuple[np.ndarray, np.ndarray]:
    if likelihood == "skew_normal":
        y1 = skewnorm.pdf(x, a=p1["alpha"], loc=p1["mu"], scale=max(p1["sigma"], 1e-12))
        y2 = skewnorm.pdf(x, a=p2["alpha"], loc=p2["mu"], scale=max(p2["sigma"], 1e-12))
        return y1, y2
    return _likelihood_pdf_from_posterior(likelihood=likelihood, x=x, params_1=p1, params_2=p2)


def support_upper_from_meta(meta: dict, feature: str, default: float = 1.0) -> float:
    cfg = meta.get("config") or {}
    param_sel = cfg.get("parameter_selection") or {}
    spec = param_sel.get(feature) or {}
    if isinstance(spec, dict) and "support_upper" in spec:
        return float(spec["support_upper"])
    return float(default)


def scale_observations_for_likelihood(
    observed_2d: np.ndarray,
    *,
    likelihood: str,
    support_upper: float,
) -> np.ndarray:
    """Match model observation scale (e.g. range / support_upper → (0,1))."""
    arr = np.asarray(observed_2d, dtype=float)
    if likelihood in {"beta", "beta_constrained", "interval_inflated_beta", "zero_inflated_beta"}:
        return arr / float(support_upper)
    return arr


def plot_likelihood_mean(
    trace,
    pairs: dict,
    observed_2d: np.ndarray,
    likelihood: str,
    feature: str,
    out_dir: Path,
    title: str,
    *,
    threshold: float = 0.9,
    support_upper: float = 1.0,
) -> None:
    setup_style(11)
    tau_map = _tau_map_from_trace(trace)
    obs_scaled = scale_observations_for_likelihood(
        observed_2d, likelihood=likelihood, support_upper=support_upper
    )
    obs_before, obs_after = _observed_split_by_tau(obs_scaled, tau_map, likelihood=likelihood)
    observed_flat = np.asarray(obs_scaled, dtype=float).ravel()
    x = _profile_x_grid(observed_flat, likelihood=likelihood, grid_size=300)
    p1, p2 = _mean_params(trace, pairs, likelihood, threshold=threshold)
    y_before, y_after = _mean_pdf(likelihood, x, p1, p2)

    if feature == "mean":
        xlab = "normalized daily REM mean"
    elif feature == "range":
        xlab = rf"daily REM range / {support_upper:g}  (unit interval)"
    else:
        xlab = f"normalized {feature}"

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if obs_before.size:
        ax.hist(
            obs_before,
            bins=18,
            density=True,
            alpha=0.28,
            color=COLOR_BEFORE,
            edgecolor="white",
            label=rf"obs before $\tau_{{\mathrm{{MAP}}}}={tau_map}$",
        )
    if obs_after.size:
        ax.hist(
            obs_after,
            bins=18,
            density=True,
            alpha=0.28,
            color=COLOR_AFTER,
            edgecolor="white",
            label=r"obs after $\tau_{\mathrm{MAP}}$",
        )
    ax.plot(x, y_before, color=COLOR_BEFORE, lw=2.4, label=f"{likelihood} @ mean $\\theta_1$")
    ax.plot(x, y_after, color=COLOR_AFTER, lw=2.4, label=f"{likelihood} @ mean $\\theta_2$")
    if likelihood == "interval_inflated_beta":
        ax.axvline(threshold, color="#888888", ls=":", lw=1.2, label=f"IIB threshold={threshold:g}")
        ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlab)
    ax.set_ylabel("density")
    ax.set_title(f"Likelihood at posterior-mean parameters — {title}", loc="left", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_dir, "04_likelihood_mean")


def plot_likelihood_fan(
    trace,
    pairs: dict,
    observed_2d: np.ndarray,
    likelihood: str,
    feature: str,
    out_dir: Path,
    title: str,
    *,
    n_curves: int = 400,
    threshold: float = 0.9,
    support_upper: float = 1.0,
) -> None:
    setup_style(11)
    tau_map = _tau_map_from_trace(trace)
    obs_scaled = scale_observations_for_likelihood(
        observed_2d, likelihood=likelihood, support_upper=support_upper
    )
    obs_before, obs_after = _observed_split_by_tau(obs_scaled, tau_map, likelihood=likelihood)
    observed_flat = np.asarray(obs_scaled, dtype=float).ravel()
    x = _profile_x_grid(observed_flat, likelihood=likelihood, grid_size=300)

    n_total = len(_values_flat(trace, next(v for v in active_trace_vars(pairs))))
    rng = np.random.default_rng(0)
    idx = rng.choice(n_total, size=min(n_curves, n_total), replace=False)

    draws_b = _draw_arrays_regime(trace, pairs, likelihood, 1, idx)
    draws_a = _draw_arrays_regime(trace, pairs, likelihood, 2, idx)
    pdf_b = _pdf_batch(likelihood, x, draws_b, threshold=threshold)
    pdf_a = _pdf_batch(likelihood, x, draws_a, threshold=threshold)

    if feature == "mean":
        xlab = "normalized daily REM mean"
    elif feature == "range":
        xlab = rf"daily REM range / {support_upper:g}  (unit interval)"
    else:
        xlab = f"normalized {feature}"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, pdfs, obs, color, lab in (
        (axes[0], pdf_b, obs_before, COLOR_BEFORE, "before"),
        (axes[1], pdf_a, obs_after, COLOR_AFTER, "after"),
    ):
        med = np.median(pdfs, axis=0)
        lo90, hi90 = np.percentile(pdfs, [5, 95], axis=0)
        step = max(1, len(pdfs) // 120)
        for row in pdfs[::step]:
            ax.plot(x, row, color=color, alpha=0.04, lw=0.8)
        ax.fill_between(x, lo90, hi90, color=color, alpha=0.18, label="90% pdf band")
        ax.plot(x, med, color=color, lw=2.3, label="median pdf")
        if obs.size:
            ax.hist(obs, bins=16, density=True, alpha=0.22, color=color, edgecolor="white", label="observations")
        if likelihood == "interval_inflated_beta":
            ax.axvline(threshold, color="#888888", ls=":", lw=1.1)
            ax.set_xlim(0.0, 1.0)
        ax.set_title(rf"{lab} $\tau$ (MAP={tau_map})", loc="left", fontweight="bold")
        ax.set_xlabel(xlab)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("density")
    fig.suptitle(f"Likelihood fan (posterior draws) — {title}", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "05_likelihood_fan")


def plot_likelihood_overlay_bands(
    trace,
    pairs: dict,
    observed_2d: np.ndarray,
    likelihood: str,
    feature: str,
    out_dir: Path,
    title: str,
    *,
    n_curves: int = 400,
    threshold: float = 0.9,
    support_upper: float = 1.0,
) -> None:
    """Like fig 5 (median + 90% band) but before/after on one axis as in fig 4.

    Thin posterior-draw curves are omitted on purpose (too much visual noise).
    """
    setup_style(11)
    tau_map = _tau_map_from_trace(trace)
    obs_scaled = scale_observations_for_likelihood(
        observed_2d, likelihood=likelihood, support_upper=support_upper
    )
    obs_before, obs_after = _observed_split_by_tau(obs_scaled, tau_map, likelihood=likelihood)
    observed_flat = np.asarray(obs_scaled, dtype=float).ravel()
    x = _profile_x_grid(observed_flat, likelihood=likelihood, grid_size=300)

    n_total = len(_values_flat(trace, next(v for v in active_trace_vars(pairs))))
    rng = np.random.default_rng(0)
    idx = rng.choice(n_total, size=min(n_curves, n_total), replace=False)

    draws_b = _draw_arrays_regime(trace, pairs, likelihood, 1, idx)
    draws_a = _draw_arrays_regime(trace, pairs, likelihood, 2, idx)
    pdf_b = _pdf_batch(likelihood, x, draws_b, threshold=threshold)
    pdf_a = _pdf_batch(likelihood, x, draws_a, threshold=threshold)

    if feature == "mean":
        xlab = "normalized daily REM mean"
    elif feature == "range":
        xlab = rf"daily REM range / {support_upper:g}  (unit interval)"
    else:
        xlab = f"normalized {feature}"

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if obs_before.size:
        ax.hist(
            obs_before,
            bins=18,
            density=True,
            alpha=0.22,
            color=COLOR_BEFORE,
            edgecolor="white",
            label=rf"obs before $\tau_{{\mathrm{{MAP}}}}={tau_map}$",
        )
    if obs_after.size:
        ax.hist(
            obs_after,
            bins=18,
            density=True,
            alpha=0.22,
            color=COLOR_AFTER,
            edgecolor="white",
            label=r"obs after $\tau_{\mathrm{MAP}}$",
        )

    for pdfs, color, lab in (
        (pdf_b, COLOR_BEFORE, r"before $\tau$"),
        (pdf_a, COLOR_AFTER, r"after $\tau$"),
    ):
        med = np.median(pdfs, axis=0)
        lo90, hi90 = np.percentile(pdfs, [5, 95], axis=0)
        ax.fill_between(x, lo90, hi90, color=color, alpha=0.18, label=f"{lab} 90% band")
        ax.plot(x, med, color=color, lw=2.4, label=f"{lab} median pdf")

    if likelihood == "interval_inflated_beta":
        ax.axvline(threshold, color="#888888", ls=":", lw=1.2, label=f"IIB threshold={threshold:g}")
        ax.set_xlim(0.0, 1.0)

    ax.set_xlabel(xlab)
    ax.set_ylabel("density")
    ax.set_title(
        f"Likelihood overlay (median + 90% band) — {title}",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "06_likelihood_overlay")


def write_readme(out_dir: Path, meta: dict, feature: str, likelihood: str, pairs: dict) -> None:
    text = f"""# Refit diagnostics

| Field | Value |
|-------|-------|
| fingerprint | `{meta.get("fingerprint")}` |
| features | `{meta.get("features")}` |
| likelihoods | `{meta.get("likelihoods")}` |
| active feature / lik | `{feature}` / `{likelihood}` |
| n_points / overlap | `{meta.get("n_points")}` / `{meta.get("overlap")}` |
| n_model_events | `{meta.get("n_model_events")}` |
| MCMC | tune={meta.get("tune")}, draws={meta.get("draws")}, chains={meta.get("chains")} |
| tau_map | `{meta.get("tau_map")}` (conc. {float(meta.get("tau_map_concentration", float("nan"))):.3f}) |

Parameter pairs: `{pairs}`

## Figures
| File | Content |
|------|---------|
| `01_traces.png` | MCMC traces |
| `02_params_overlay.png` | Before/after parameter KDEs + P(τ) |
| `03_tau_posterior.png` | Onset posterior |
| `04_likelihood_mean.png` | Density at posterior-mean parameters |
| `05_likelihood_fan.png` | Fan + 90% pdf band (two panels) |
| `06_likelihood_overlay.png` | Median + 90% band, before/after on one axis (no thin curves) |
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def resolve_model_dir(refit_dir: Path) -> Path:
    if (refit_dir / "trace.nc").is_file() or (refit_dir / "trace.zarr").exists():
        return refit_dir
    candidates = sorted(refit_dir.glob("rank*_*/trace.nc"))
    if candidates:
        return candidates[0].parent
    zarr_cands = sorted(refit_dir.glob("rank*_*/trace.zarr"))
    if zarr_cands:
        return zarr_cands[0].parent
    raise FileNotFoundError(f"No trace.nc / trace.zarr under {refit_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refit-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-fan-curves", type=int, default=400)
    p.add_argument("--iib-threshold", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.refit_dir)
    out_dir = args.out_dir
    print(f"[load] model_dir={model_dir}")

    trace = load_trace(model_dir)
    meta = load_meta(model_dir)
    obs = load_observations(model_dir)
    feature, likelihood = parse_likelihood(meta)
    pairs = discover_param_pairs(trace, feature)
    obs_key = discover_obs_key(obs, feature)
    observed_2d = obs[obs_key]
    support_upper = support_upper_from_meta(meta, feature, default=1.0)
    # Prefer threshold from config when present
    threshold = float(args.iib_threshold)
    cfg_spec = ((meta.get("config") or {}).get("parameter_selection") or {}).get(feature) or {}
    if isinstance(cfg_spec, dict) and "threshold" in cfg_spec:
        threshold = float(cfg_spec["threshold"])
    title = f"{meta.get('fingerprint')} | {meta.get('features')} | {meta.get('likelihoods')}"
    print(f"[lik] feature={feature} likelihood={likelihood}")
    print(f"[params] {pairs}")
    print(f"[obs] key={obs_key} shape={observed_2d.shape} support_upper={support_upper}")

    # Validate required pairs
    needed = {
        "student_t": ["mu", "sigma", "nu"],
        "skew_normal": ["mu", "sigma", "alpha"],
        "interval_inflated_beta": ["alpha", "beta", "pi"],
    }.get(likelihood)
    if needed is None:
        raise ValueError(f"Unsupported likelihood for diagnostics: {likelihood}")
    for k in needed:
        if not pairs.get(k):
            raise ValueError(f"Missing parameter {k} for {likelihood}; available pairs={pairs}")
    # student_t: nu may be shared (str) or per-regime (tuple)

    plot_traces_compact(trace, active_trace_vars(pairs), out_dir, title)
    plot_params_overlay(trace, pairs, out_dir, title)
    plot_likelihood_mean(
        trace,
        pairs,
        observed_2d,
        likelihood,
        feature,
        out_dir,
        title,
        threshold=threshold,
        support_upper=support_upper,
    )
    plot_likelihood_fan(
        trace,
        pairs,
        observed_2d,
        likelihood,
        feature,
        out_dir,
        title,
        n_curves=int(args.n_fan_curves),
        threshold=threshold,
        support_upper=support_upper,
    )
    plot_likelihood_overlay_bands(
        trace,
        pairs,
        observed_2d,
        likelihood,
        feature,
        out_dir,
        title,
        n_curves=int(args.n_fan_curves),
        threshold=threshold,
        support_upper=support_upper,
    )
    write_readme(out_dir, meta, feature, likelihood, pairs)
    print("[done]")


if __name__ == "__main__":
    main()
