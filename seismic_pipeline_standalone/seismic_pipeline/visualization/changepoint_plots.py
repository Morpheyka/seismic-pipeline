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

from seismic_pipeline.bayesian.diagnostics import (
    _available_varnames,
    _is_inferencedata,
    _observed_split_by_tau,
    _tau_map_from_trace,
    _values_by_chain,
    _values_flat,
    tau_probabilities,
)
from seismic_pipeline.bayesian.search import compute_model_distance_matrix


def _show_or_save(save_path: str | os.PathLike[str] | None) -> None:
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_trace_and_tau(
    trace,
    trace_vars,
    title_prefix: str,
    *,
    save_path: str | os.PathLike[str] | None = None,
):
    """Plot traces for selected vars and tau bar chart."""
    if _is_inferencedata(trace):
        az.plot_trace(trace, var_names=trace_vars)
    else:
        posterior = {}
        for var in trace_vars:
            chains = _values_by_chain(trace, var)
            posterior[var] = np.stack(chains, axis=0)
        idata = az.from_dict({"posterior": posterior})
        az.plot_trace(idata, var_names=trace_vars)
    plt.suptitle(f"Trace plots: {title_prefix}", y=1.02)
    plt.tight_layout()
    _show_or_save(save_path)


def plot_posteriors_like_script(
    trace,
    group_data: dict,
    title_prefix: str,
    *,
    save_path: str | os.PathLike[str] | None = None,
    tau_bar_save_path: str | os.PathLike[str] | None = None,
):
    """Posterior histograms for before/after tau in script-like style."""
    trace_vars = _available_varnames(trace)
    pairs = []
    for group_name, features in group_data.items():
        for feat_name in features:
            for param_name in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in trace_vars and p2 in trace_vars:
                    pairs.append(((p1, p2), f"{group_name} {feat_name} {param_name}"))

    n_axes = len(pairs) + 1
    n_cols = 3
    n_rows = (n_axes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    color_before, color_after = "#A60628", "#7A68A6"
    for i, ((name_1, name_2), title) in enumerate(pairs):
        ax = axes[i]
        # `edgecolor/linewidth` adds visible "borders" around histogram bins.
        ax.hist(
            _values_flat(trace, name_1),
            bins=30,
            alpha=0.6,
            density=True,
            color=color_before,
            edgecolor="black",
            linewidth=0.6,
            label="до tau",
        )
        ax.hist(
            _values_flat(trace, name_2),
            bins=30,
            alpha=0.6,
            density=True,
            color=color_after,
            edgecolor="black",
            linewidth=0.6,
            label="после tau",
        )
        ax.set_title(title)
        ax.legend()

    tau_idx = len(pairs)
    support, probs = tau_probabilities(trace)
    axes[tau_idx].bar(support, probs, color="#348ABD", width=0.8, edgecolor="black", linewidth=0.6)
    axes[tau_idx].set_xticks(support)
    axes[tau_idx].set_title("tau")
    axes[tau_idx].set_xlabel("индекс группы чанков")

    for j in range(tau_idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Постериоры (до/после tau): {title_prefix}", y=1.02)
    plt.tight_layout()
    _show_or_save(save_path)

    support, probs = tau_probabilities(trace)
    plt.figure(figsize=(6, 3.5))
    plt.bar(support, probs, width=0.8)
    plt.xticks(support)
    plt.xlabel("tau (индекс группы чанков)")
    plt.ylabel("P(tau=k)")
    plt.title(f"Постериорное распределение tau: {title_prefix}")
    plt.tight_layout()
    _show_or_save(tau_bar_save_path)
def plot_exhaustive_search_results(
    results: list[dict],
    *,
    top_n: int = 20,
    pareto_threshold: float = 0.7,
) -> None:
    """Plot diagnostics for exhaustive model search outputs."""
    ok = [
        r for r in results
        if r.get("status") == "ok"
        and math.isfinite(float(r.get("elpd_loo_per_feature", float("nan"))))
    ]
    if not ok:
        print("No successful exhaustive-search records to plot.")
        return

    sorted_res = sorted(
        ok,
        key=lambda x: float(x.get("elpd_loo_per_feature", float("-inf"))),
        reverse=True,
    )
    top = sorted_res[: max(1, int(top_n))]

    loo_vals = np.asarray(
        [float(r.get("elpd_loo_per_feature", float("nan"))) for r in sorted_res],
        dtype=float,
    )
    e_tau_vals = np.asarray([float(r.get("e_tau", float("nan"))) for r in sorted_res], dtype=float)
    tau_std_vals = np.asarray([float(r.get("tau_std", float("nan"))) for r in sorted_res], dtype=float)
    n_blocks = np.asarray([int(r.get("n_feature_blocks", 0)) for r in sorted_res], dtype=float)
    ess_vals = np.asarray([float(r.get("ess_min_bulk", float("nan"))) for r in sorted_res], dtype=float)
    pareto_k_vals = np.asarray([float(r.get("loo_pareto_k_max", float("nan"))) for r in sorted_res], dtype=float)
    idx = np.arange(1, len(sorted_res) + 1, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(
        idx,
        loo_vals,
        "o-",
        ms=4,
        lw=1.1,
        color="#1f77b4",
        label="elpd_loo / (n_features * n_events)",
    )
    ax.axhline(
        float(np.nanmax(loo_vals)),
        color="#2ca02c",
        ls="--",
        lw=1.0,
        label="best elpd_loo / (n_features * n_events)",
    )
    ax.invert_xaxis()
    ax.set_xlabel("Model rank (by elpd_loo / (n_features * n_events), 1 = best)")
    ax.set_ylabel("elpd_loo / (n_features * n_events)")
    ax.set_title("Normalized elpd_loo vs model rank")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    top_cfgs = [r.get("config") or {} for r in top]
    dist = compute_model_distance_matrix(top_cfgs)
    im = ax.imshow(dist, cmap="viridis", aspect="auto")
    labels = [str(r.get("fingerprint", ""))[:8] for r in top]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Hamming distance heatmap (top {len(top)})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    valid_tau = np.isfinite(loo_vals) & np.isfinite(e_tau_vals) & np.isfinite(tau_std_vals)
    n_skipped_tau = int(np.size(valid_tau) - int(np.count_nonzero(valid_tau)))
    if n_skipped_tau > 0:
        warnings.warn(
            f"plot_exhaustive_search_results: skipped {n_skipped_tau} models with missing e_tau/tau_std.",
            UserWarning,
            stacklevel=2,
        )
    sc = None
    if np.any(valid_tau):
        ax.errorbar(
            loo_vals[valid_tau],
            e_tau_vals[valid_tau],
            yerr=tau_std_vals[valid_tau],
            fmt="none",
            ecolor="gray",
            alpha=0.4,
            capsize=2,
            linewidth=0.5,
        )
        sc = ax.scatter(
            loo_vals[valid_tau],
            e_tau_vals[valid_tau],
            c=n_blocks[valid_tau],
            cmap="plasma",
            s=36,
            alpha=0.85,
            edgecolors="none",
        )
    ax.set_xlabel("elpd_loo per feature")
    ax.set_ylabel("E[tau] (days)")
    ax.set_title("elpd_loo per feature vs E[tau] (bars = +/-1 std)")
    ax.grid(alpha=0.3)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("n_feature_blocks")

    parent_ax = axes[1, 1]
    parent_spec = parent_ax.get_subplotspec()
    parent_ax.remove()
    sub_axes = parent_spec.subgridspec(1, 2, wspace=0.4)
    ax_ess = fig.add_subplot(sub_axes[0, 0])
    ax_pk = fig.add_subplot(sub_axes[0, 1])

    finite_ess = ess_vals[np.isfinite(ess_vals)]
    if finite_ess.size:
        ax_ess.hist(
            finite_ess,
            bins=min(20, max(5, finite_ess.size // 2)),
            alpha=0.55,
            label="ess_min_bulk",
            color="#1f77b4",
        )
    ax_ess.axvline(100.0, color="#2ca02c", ls="--", lw=1.0, label="ESS threshold")
    ax_ess.set_title("ESS distribution")
    ax_ess.set_xlabel("Value")
    ax_ess.set_ylabel("Count")
    ax_ess.grid(alpha=0.3)
    ax_ess.legend(loc="best", fontsize=8)

    finite_pareto = pareto_k_vals[np.isfinite(pareto_k_vals)]
    if finite_pareto.size:
        ax_pk.hist(
            finite_pareto,
            bins=20,
            color="orange",
            edgecolor="black",
            alpha=0.7,
        )
    ax_pk.axvline(0.5, color="red", ls="--", lw=1.0, label="k=0.5 (reliable)")
    ax_pk.axvline(0.7, color="darkred", ls=":", lw=1.0, label="k=0.7 (unreliable)")
    n_bad_pareto = int(np.count_nonzero(finite_pareto >= pareto_threshold))
    ax_pk.text(
        0.95,
        0.95,
        f"k >= {pareto_threshold:g}: {n_bad_pareto} models",
        transform=ax_pk.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax_pk.set_xlabel("max Pareto k")
    ax_pk.set_ylabel("N models")
    ax_pk.set_title("Pareto-k distribution")
    ax_pk.grid(alpha=0.3)
    ax_pk.legend(loc="best", fontsize=7)

    plt.suptitle(
        f"Exhaustive search diagnostics (models with Pareto-k > {pareto_threshold:g}: {n_bad_pareto})",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
def plot_model_search_results(
    search_result: dict,
    summary: dict[str, Any] | None = None,
    *,
    title: str = "Metropolis-Hastings model search",
) -> None:
    """Plot MH chain and visit shares for features, groups, n_chunks, and likelihood families."""
    chain = list(search_result.get("chain") or [])
    if not chain:
        print("No MH chain to plot.")
        return

    summary = summary if summary is not None else summarize_model_search(search_result)

    iterations = np.array([int(r["iteration"]) for r in chain], dtype=float)
    log_targets = np.array([float(r.get("log_target", float("nan"))) for r in chain], dtype=float)
    accepted = np.array([bool(r.get("accepted", False)) for r in chain], dtype=bool)
    q10_arr = np.array([float(r.get("log_target_q10", float("nan"))) for r in chain], dtype=float)
    q50_arr = np.array([float(r.get("log_target_q50", float("nan"))) for r in chain], dtype=float)
    q90_arr = np.array([float(r.get("log_target_q90", float("nan"))) for r in chain], dtype=float)

    thr = float(chain[0].get("config", {}).get("tau_threshold", 7.0))
    p_gt: List[float] = []
    map_tau: List[float] = []
    bfmi_arr: List[float] = []
    ndiv_arr: List[float] = []
    for r in chain:
        sc = r.get("score")
        if isinstance(sc, dict):
            p_gt.append(float(sc.get("p_tau_gt_threshold", float("nan"))))
            map_tau.append(float(sc.get("map_tau", float("nan"))))
            b = sc.get("bfmi", sc.get("bfmi_approx"))
            if b is not None and math.isfinite(float(b)):
                bfmi_arr.append(float(b))
            else:
                bfmi_arr.append(float("nan"))
            ndiv_arr.append(float(sc.get("n_divergences", 0)))
        else:
            p_gt.append(float("nan"))
            map_tau.append(float("nan"))
            bfmi_arr.append(float("nan"))
            ndiv_arr.append(float("nan"))
    p_gt_arr = np.asarray(p_gt, dtype=float)
    map_tau_arr = np.asarray(map_tau, dtype=float)
    bfmi_plot = np.asarray(bfmi_arr, dtype=float)
    ndiv_plot = np.asarray(ndiv_arr, dtype=float)

    full_title = title
    flt = summary.get("final_log_targets") or []
    flstd = summary.get("final_log_target_std")
    if len(flt) > 1:
        try:
            sd = float(flstd)
            if math.isfinite(sd):
                full_title = f"{title} (std of final log-target across {len(flt)} MH chains: {sd:.4g})"
        except (TypeError, ValueError):
            pass

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax_lt = axes[0, 0]
    if np.any(np.isfinite(q10_arr)) and np.any(np.isfinite(q90_arr)):
        ax_lt.fill_between(
            iterations,
            q10_arr,
            q90_arr,
            color="#348ABD",
            alpha=0.22,
            label="log-target q10–q90 (rolling)",
        )
        ax_lt.plot(
            iterations,
            q50_arr,
            "--",
            color="#1f77b4",
            lw=1.2,
            alpha=0.85,
            label="rolling median",
        )
    ax_lt.plot(iterations, log_targets, "o-", color="#E24A33", ms=5, lw=1.2, label="current log-target")
    ax_lt.set_xlabel("MH iteration")
    ax_lt.set_ylabel("log target (current state)")
    ax_lt.set_title("Log-target along chain (+ rolling quantiles)")
    ax_lt.grid(alpha=0.3)
    ax_lt.legend(loc="best", fontsize=8)

    if len(chain) > 1:
        acc_float = accepted[1:].astype(float)
        run_mean = np.cumsum(acc_float) / np.arange(1, len(acc_float) + 1, dtype=float)
        axes[0, 1].plot(iterations[1:], run_mean, color="#E24A33", lw=1.5)
        axes[0, 1].scatter(iterations[1:], acc_float, c=["#2ca02c" if a else "#7f7f7f" for a in accepted[1:]], s=22, zorder=3)
        axes[0, 1].set_xlabel("MH iteration")
        axes[0, 1].set_ylabel("cumulative P(accept proposal)")
        axes[0, 1].set_title("Running acceptance rate (green=accepted)")
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].grid(alpha=0.3)
    else:
        axes[0, 1].axis("off")

    axb = axes[1, 0]
    axb.plot(iterations, p_gt_arr, "s-", color="#2ca02c", ms=4, lw=1, label=rf"$P(\tau > {thr:g})$")
    axb.set_xlabel("MH iteration")
    axb.set_ylabel(rf"$P(\tau > {thr:g})$")
    axb.set_ylim(-0.05, 1.05)
    axb.legend(loc="upper left")
    axb.grid(alpha=0.3)

    ax2 = axb.twinx()
    ax2.plot(iterations, map_tau_arr, "D--", color="#9467bd", ms=4, lw=1, alpha=0.85, label="MAP " + r"$\tau$")
    ax2.set_ylabel("MAP " + r"$\tau$" + " (from posterior over support)")
    ax2.legend(loc="upper right")

    axb.set_title(r"$\tau$ signal along chain (marginalized / discrete)")

    ax_s = axes[1, 1]
    ax_s.plot(iterations, bfmi_plot, "o-", color="#8c564b", ms=4, lw=1.1, label="BFMI")
    ax_s.axhline(0.3, color="gray", ls="--", lw=0.9, label="BFMI=0.3")
    ax_s.set_xlabel("MH iteration")
    ax_s.set_ylabel("BFMI")
    ax_s.set_title("Sampler diagnostics (current model state)")
    ax_s.grid(alpha=0.3)
    ax_sd = ax_s.twinx()
    ax_sd.bar(
        iterations,
        ndiv_plot,
        color="#E24A33",
        width=0.8,
        alpha=0.9,
        label="NUTS divergences / fit",
    )
    ax_sd.set_ylabel("divergences")
    ax_s.legend(loc="upper left", fontsize=8)
    ax_sd.legend(loc="upper right", fontsize=8)

    plt.suptitle(full_title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

    feat = summary.get("feature_visit_freq") or {}
    if feat:
        names = list(feat.keys())
        vals = [float(feat[k]) for k in names]
        y_pos = np.arange(len(names))
        fig_feat, axf = plt.subplots(figsize=(6.5, max(2.5, 0.35 * len(names))))
        axf.barh(y_pos, vals, color="#A60628", height=0.65)
        axf.set_yticks(y_pos)
        axf.set_yticklabels(names)
        axf.set_xlabel("visit share")
        axf.set_title("Feature metrics in model (visit share)")
        axf.set_xlim(0, 1.05)
        axf.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    grp = summary.get("group_visit_freq") or {}
    if grp:
        names = list(grp.keys())
        vals = [float(grp[k]) for k in names]
        y_pos = np.arange(len(names))
        fig_grp, axg = plt.subplots(figsize=(6.5, max(2.3, 0.35 * len(names))))
        axg.barh(y_pos, vals, color="#348ABD", height=0.65)
        axg.set_yticks(y_pos)
        axg.set_yticklabels(names)
        axg.set_xlabel("visit share")
        axg.set_title("Feature groups in model (visit share)")
        axg.set_xlim(0, 1.05)
        axg.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    n_chunks_freq = summary.get("n_chunks_visit_freq") or {}
    if n_chunks_freq:
        labels = list(n_chunks_freq.keys())
        vals = [float(n_chunks_freq[k]) for k in labels]
        fig_nc, axn = plt.subplots(figsize=(max(5.0, 0.9 * len(labels)), 3.1))
        axn.bar(labels, vals, color="#2ca02c", edgecolor="black", linewidth=0.6)
        axn.set_xlabel("n_chunks")
        axn.set_ylabel("visit share")
        axn.set_title("n_chunks in model (visit share)")
        axn.set_ylim(0, 1.05)
        axn.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    lk = summary.get("likelihood_visit_freq") or {}
    if not lk:
        return

    metrics_sorted = sorted(lk.keys())
    n_m = len(metrics_sorted)
    fig2, axes2 = plt.subplots(1, n_m, figsize=(max(4.0 * n_m, 5), 3.2), squeeze=False)
    for i, metric in enumerate(metrics_sorted):
        ax = axes2[0, i]
        dists = lk[metric]
        labs = list(dists.keys())
        vs = [float(dists[lab]) for lab in labs]
        ax.bar(labs, vs, color="#7A68A6", edgecolor="black", linewidth=0.5)
        ax.set_title(metric)
        ax.set_ylabel("visit share")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Likelihood family (visit share per metric)", fontsize=11, y=1.08)
    plt.tight_layout()
    plt.show()


def plot_observations_before_after(
    *,
    group_name: str,
    feat_name: str,
    likelihood: str,
    observed_2d: np.ndarray,
    trace,
    title_prefix: str = "",
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Histogram of raw observations split by MAP tau (before vs after)."""
    tau_map = _tau_map_from_trace(trace)
    obs_before, obs_after = _observed_split_by_tau(
        observed_2d,
        tau_map,
        likelihood=likelihood,
    )
    color_before, color_after = "#A60628", "#7A68A6"
    plt.figure(figsize=(7, 4))
    if obs_before.size > 0:
        plt.hist(
            obs_before,
            bins=15,
            density=True,
            alpha=0.55,
            color=color_before,
            edgecolor="black",
            linewidth=0.5,
            label="наблюдения до tau",
        )
    if obs_after.size > 0:
        plt.hist(
            obs_after,
            bins=15,
            density=True,
            alpha=0.55,
            color=color_after,
            edgecolor="black",
            linewidth=0.5,
            label="наблюдения после tau",
        )
    title = f"Наблюдения: {group_name}/{feat_name}"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _show_or_save(save_path)


def plot_feature_likelihood_profile(
    *,
    group_name: str,
    feat_name: str,
    likelihood: str,
    x: np.ndarray,
    y_before: np.ndarray,
    y_after: np.ndarray,
    observed_2d: np.ndarray,
    trace,
    use_log_y: bool = False,
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Plot before/after likelihood profile for one feature."""
    plt.figure(figsize=(7, 4))
    tau_map = _tau_map_from_trace(trace)
    obs_before, obs_after = _observed_split_by_tau(
        observed_2d,
        tau_map,
        likelihood=likelihood,
    )
    color_before, color_after = "#A60628", "#7A68A6"
    if obs_before.size > 0:
        plt.hist(
            obs_before,
            bins=15,
            density=True,
            alpha=0.25,
            color=color_before,
            label="observations before tau",
        )
    if obs_after.size > 0:
        plt.hist(
            obs_after,
            bins=15,
            density=True,
            alpha=0.25,
            color=color_after,
            label="observations after tau",
        )
    y_plot_before = np.maximum(y_before, 1e-300) if use_log_y else y_before
    y_plot_after = np.maximum(y_after, 1e-300) if use_log_y else y_after
    plt.plot(x, y_plot_before, color="#A60628", linewidth=2.0, label="before tau")
    plt.plot(x, y_plot_after, color="#7A68A6", linewidth=2.0, label="after tau")
    if use_log_y:
        plt.yscale("log")
    plt.title(f"Likelihood profile: {group_name}/{feat_name} ({likelihood})")
    plt.xlabel("value")
    plt.ylabel("density (log scale)" if use_log_y else "density")
    plt.grid(alpha=0.25, which="both" if use_log_y else "major")
    plt.legend()
    plt.tight_layout()
    _show_or_save(save_path)


def plot_interval_inflated_beta_pdf_comparison(
    *,
    group_name: str,
    feat_name: str,
    x: np.ndarray,
    y_before_model: np.ndarray,
    y_after_model: np.ndarray,
    y_before_mixture: np.ndarray,
    y_after_mixture: np.ndarray,
    threshold: float,
    use_log_y: bool = False,
    save_path: str | os.PathLike[str] | None = None,
) -> None:
    """Compare interval_inflated_beta PDF: piecewise model vs legacy mixture plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    panels = (
        ("before tau", y_before_model, y_before_mixture, "#A60628"),
        ("after tau", y_after_model, y_after_mixture, "#7A68A6"),
    )
    for ax, (title, y_model, y_mix, color) in zip(axes, panels):
        y_model_plot = np.maximum(y_model, 1e-300) if use_log_y else y_model
        y_mix_plot = np.maximum(y_mix, 1e-300) if use_log_y else y_mix
        ax.plot(x, y_model_plot, color=color, linewidth=2.2, label="model (piecewise)")
        ax.plot(
            x,
            y_mix_plot,
            color=color,
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label="legacy mixture plot",
        )
        ax.axvline(float(threshold), color="gray", linestyle=":", linewidth=1.2, label=f"threshold={threshold:g}")
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.grid(alpha=0.25, which="both" if use_log_y else "major")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density (log scale)" if use_log_y else "density")
    if use_log_y:
        for ax in axes:
            ax.set_yscale("log")
    fig.suptitle(
        f"IIB PDF comparison: {group_name}/{feat_name}\n"
        "model: uniform on [t,1] above t, beta below; mixture: pi*uniform + (1-pi)*beta everywhere",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout()
    _show_or_save(save_path)
