#!/usr/bin/env python3
"""Null-control suite for primary mean-only changepoint arm.

Experiments (run selectively via --exps):
  4  synthetic single-CP power check (known shared onset → within-window shuffle)
  3  i.i.d. pooled-value null on real observations
  1  many within-window seeds: peak / E[τ] distribution (not mean-PMF only)
  2  column (day-index) shuffle across windows

Usage:
  ../.venv/bin/python scripts/run_null_control_suite.py --exps 4
  ../.venv/bin/python scripts/run_null_control_suite.py --exps 4,3,1,2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import score_changepoint_trace, tau_probabilities

DEFAULT_REAL_DIR = (
    PROJECT_ROOT
    / "run_output_8day_density_safe"
    / "refit_best_mean_only"
    / "rank11_4f6e4c855d72864d"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "run_output_8day_density_safe" / "null_control_suite"

FEATURE_SELECTION = {"daily": ["mean"]}
PARAMETER_SELECTION = {"mean": {"likelihood": "student_t"}}
TAU_LOWER = 2
TAU_UPPER = 8
FINGERPRINT = "4f6e4c855d72864d"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--exps", default="4,3,1,2", help="Comma list from {1,2,3,4}")
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--tune-many", type=int, default=1000, help="MCMC tune for exp 1")
    p.add_argument("--draws-many", type=int, default=500, help="MCMC draws for exp 1")
    p.add_argument("--chains-many", type=int, default=2)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--many-seeds", default="", help="Override for exp1, e.g. 0-19")
    p.add_argument("--nuts-backend", default="blackjax")
    p.add_argument("--synth-tau", type=int, default=6)
    p.add_argument("--synth-n-windows", type=int, default=33)
    p.add_argument("--synth-mu1", type=float, default=-0.55)
    p.add_argument("--synth-mu2", type=float, default=-0.25)
    p.add_argument("--synth-sigma", type=float, default=0.08)
    return p.parse_args()


def _parse_seed_list(spec: str) -> list[int]:
    """Parse '0,1,2' or '0-19,25' into sorted unique ints."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if part.count("-") == 1:
            a, b = part.split("-")
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                out.extend(range(int(a), int(b) + 1))
                continue
        out.append(int(part))
    return sorted(set(out))


def _load_observations(real_dir: Path) -> np.ndarray:
    path = real_dir / "observations.npz"
    z = np.load(path)
    return np.asarray(z["daily__mean"], dtype=float)


def within_window_day_shuffle(obs: np.ndarray, *, seed: int) -> np.ndarray:
    out = np.array(obs, dtype=float, copy=True)
    rng = np.random.default_rng(int(seed))
    for i in range(out.shape[0]):
        idxs = np.flatnonzero(np.isfinite(out[i]))
        if idxs.size < 2:
            continue
        values = out[i, idxs].copy()
        out[i, idxs] = values[rng.permutation(idxs.size)]
    return out


def column_day_shuffle(obs: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute values across windows within each day column (finite cells only)."""
    out = np.array(obs, dtype=float, copy=True)
    rng = np.random.default_rng(int(seed))
    n_windows, n_days = out.shape
    for t in range(n_days):
        idxs = np.flatnonzero(np.isfinite(out[:, t]))
        if idxs.size < 2:
            continue
        values = out[idxs, t].copy()
        out[idxs, t] = values[rng.permutation(idxs.size)]
    return out


def iid_pooled_null(obs: np.ndarray, *, seed: int) -> np.ndarray:
    """Replace each finite cell with i.i.d. draw from pooled finite values."""
    out = np.array(obs, dtype=float, copy=True)
    rng = np.random.default_rng(int(seed))
    pool = out[np.isfinite(out)]
    if pool.size == 0:
        return out
    mask = np.isfinite(out)
    out[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)
    return out


def make_synthetic_single_cp(
    *,
    n_windows: int,
    n_days: int,
    tau: int,
    mu1: float,
    mu2: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Shared onset τ: days t < τ ~ N(mu1), t >= τ ~ N(mu2). Day index 1..n_days."""
    rng = np.random.default_rng(int(seed))
    # model uses 0-based columns; tau is 1-based day index with before = t < tau
    days = np.arange(1, n_days + 1)
    before = days < int(tau)
    means = np.where(before, mu1, mu2)
    obs = rng.normal(loc=means, scale=sigma, size=(n_windows, n_days))
    return obs.astype(float)


def _group_data(obs: np.ndarray) -> dict[str, dict[str, pd.DataFrame]]:
    return {"daily": {"mean": pd.DataFrame(np.asarray(obs, dtype=float))}}


def _pmf_mad_uniform(probs: list[float] | np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        return float("nan")
    u = 1.0 / float(p.size)
    return float(np.mean(np.abs(p - u)))


def _peak_k(support: list[int], probs: list[float]) -> int:
    i = int(np.argmax(np.asarray(probs, dtype=float)))
    return int(support[i])


def _fit(
    obs: np.ndarray,
    *,
    tune: int,
    draws: int,
    chains: int,
    nuts_backend: str,
) -> dict[str, Any]:
    group_data = _group_data(obs)
    model = build_changepoint_model(
        group_data,
        tau_lower=TAU_LOWER,
        tau_upper=TAU_UPPER,
        parameter_selection=PARAMETER_SELECTION,
        tau_mode="marginalized",
    )
    sample_kwargs: dict[str, Any] = dict(
        draws=int(draws),
        tune=int(tune),
        chains=int(chains),
        progressbar=False,
        nuts_backend=nuts_backend,
    )
    if nuts_backend == "pymc":
        sample_kwargs["cores"] = 1
    else:
        sample_kwargs["jax_chain_method"] = "parallel"
        sample_kwargs["jax_var_names"] = [
            "changepoint_pointwise_log_lik",
            "tau_probs",
            "tau_support",
            "tau_mean",
        ]
        sample_kwargs["materialize_posterior_vars"] = sample_kwargs["jax_var_names"]
    t0 = time.time()
    trace = sample_model(model, **sample_kwargs)
    sample_s = time.time() - t0
    scores = score_changepoint_trace(
        trace,
        group_data=group_data,
        parameter_selection=PARAMETER_SELECTION,
        model=model,
        criterion="loo",
        warn_on_fallback=False,
        loo_report="elpd",
    )
    support, probs = tau_probabilities(trace)
    support_i = [int(x) for x in np.asarray(support).ravel()]
    probs_f = [float(x) for x in np.asarray(probs, dtype=float).ravel()]
    return {
        "e_tau": float(scores.get("e_tau", float("nan"))),
        "tau_hdi_60_width": float(scores.get("tau_hdi_60_width", float("nan"))),
        "r_hat_max": float(scores.get("r_hat_max", float("nan"))),
        "ess_min_bulk": float(scores.get("ess_min_bulk", float("nan"))),
        "tau_support": support_i,
        "mean_p_tau": probs_f,
        "peak_tau": _peak_k(support_i, probs_f),
        "peak_p": float(max(probs_f)) if probs_f else float("nan"),
        "mean_abs_dev_from_uniform": _pmf_mad_uniform(probs_f),
        "sample_s": float(sample_s),
        "n_events": int(obs.shape[0]),
        "tune": int(tune),
        "draws": int(draws),
        "chains": int(chains),
    }


def _save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    flat = []
    for r in rows:
        row = {
            k: v
            for k, v in r.items()
            if k not in {"mean_p_tau", "tau_support", "std_p_tau"}
        }
        for k, p in zip(r.get("tau_support") or [], r.get("mean_p_tau") or []):
            row[f"p_tau_{k}"] = p
        flat.append(row)
    pd.DataFrame(flat).to_csv(path.with_suffix(".csv"), index=False)
    print(f"[suite] wrote {path}", flush=True)


def run_exp4(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    print("[suite] === EXP 4: synthetic single-CP power calibration ===", flush=True)
    obs = make_synthetic_single_cp(
        n_windows=int(args.synth_n_windows),
        n_days=8,
        tau=int(args.synth_tau),
        mu1=float(args.synth_mu1),
        mu2=float(args.synth_mu2),
        sigma=float(args.synth_sigma),
        seed=123,
    )
    np.savez_compressed(out_dir / "exp4_observations_synth.npz", daily__mean=obs)
    rows: list[dict[str, Any]] = []
    print("[suite] exp4 fit synth (true aligned) …", flush=True)
    rec = _fit(
        obs,
        tune=args.tune,
        draws=args.draws,
        chains=args.chains,
        nuts_backend=args.nuts_backend,
    )
    rec.update(
        {
            "exp": 4,
            "arm": "synth_aligned",
            "seed": None,
            "true_tau": int(args.synth_tau),
        }
    )
    rows.append(rec)
    print(
        f"[suite] exp4 aligned E[τ]={rec['e_tau']:.3f} peak={rec['peak_tau']} "
        f"MAD_u={rec['mean_abs_dev_from_uniform']:.4f}",
        flush=True,
    )
    seeds = _parse_seed_list(args.seeds)
    for seed in seeds:
        sh = within_window_day_shuffle(obs, seed=seed)
        np.savez_compressed(
            out_dir / f"exp4_observations_wwshuffle_seed{seed}.npz", daily__mean=sh
        )
        print(f"[suite] exp4 within-window shuffle seed={seed} …", flush=True)
        rec_s = _fit(
            sh,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            nuts_backend=args.nuts_backend,
        )
        rec_s.update(
            {
                "exp": 4,
                "arm": "synth_within_window_shuffle",
                "seed": seed,
                "true_tau": int(args.synth_tau),
            }
        )
        rows.append(rec_s)
        print(
            f"[suite] exp4 seed={seed} E[τ]={rec_s['e_tau']:.3f} peak={rec_s['peak_tau']} "
            f"MAD_u={rec_s['mean_abs_dev_from_uniform']:.4f}",
            flush=True,
        )
    shuf = [r for r in rows if r["arm"] == "synth_within_window_shuffle"]
    if shuf:
        stack = np.vstack([r["mean_p_tau"] for r in shuf])
        rows.append(
            {
                "exp": 4,
                "arm": "synth_wwshuffle_seeds_mean",
                "seeds": seeds,
                "true_tau": int(args.synth_tau),
                "tau_support": shuf[0]["tau_support"],
                "mean_p_tau": [float(x) for x in stack.mean(axis=0)],
                "mean_abs_dev_from_uniform": _pmf_mad_uniform(stack.mean(axis=0)),
                "e_tau_mean": float(np.mean([r["e_tau"] for r in shuf])),
                "peak_tau_mode": int(
                    pd.Series([r["peak_tau"] for r in shuf]).mode().iloc[0]
                ),
                "aligned_MAD": float(rows[0]["mean_abs_dev_from_uniform"]),
                "power_ok_MAD_drop": bool(
                    _pmf_mad_uniform(stack.mean(axis=0))
                    < 0.5 * float(rows[0]["mean_abs_dev_from_uniform"])
                ),
            }
        )
    _save_results(out_dir / "exp4_synthetic_power.json", rows)
    return rows


def run_exp3(args: argparse.Namespace, out_dir: Path, obs0: np.ndarray) -> list[dict[str, Any]]:
    print("[suite] === EXP 3: i.i.d. pooled-value null ===", flush=True)
    rows: list[dict[str, Any]] = []
    # real comparator from stored zarr if present
    zarr = args.real_dir / "trace.zarr"
    if zarr.exists():
        import arviz as az

        idata = az.from_zarr(str(zarr))
        support, probs = tau_probabilities(idata)
        support_i = [int(x) for x in np.asarray(support).ravel()]
        probs_f = [float(x) for x in np.asarray(probs, dtype=float).ravel()]
        rows.append(
            {
                "exp": 3,
                "arm": "real_stored",
                "seed": None,
                "e_tau": float(np.asarray(idata.posterior["tau_mean"]).mean()),
                "tau_support": support_i,
                "mean_p_tau": probs_f,
                "peak_tau": _peak_k(support_i, probs_f),
                "peak_p": float(max(probs_f)),
                "mean_abs_dev_from_uniform": _pmf_mad_uniform(probs_f),
                "source": "stored_trace.zarr",
            }
        )
    seeds = _parse_seed_list(args.seeds)
    for seed in seeds:
        iid = iid_pooled_null(obs0, seed=seed)
        np.savez_compressed(out_dir / f"exp3_observations_iid_seed{seed}.npz", daily__mean=iid)
        print(f"[suite] exp3 i.i.d. seed={seed} …", flush=True)
        rec = _fit(
            iid,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            nuts_backend=args.nuts_backend,
        )
        rec.update({"exp": 3, "arm": "iid_pooled", "seed": seed})
        rows.append(rec)
        print(
            f"[suite] exp3 seed={seed} E[τ]={rec['e_tau']:.3f} peak={rec['peak_tau']} "
            f"MAD_u={rec['mean_abs_dev_from_uniform']:.4f}",
            flush=True,
        )
    _save_results(out_dir / "exp3_iid_null.json", rows)
    return rows


def run_exp1(args: argparse.Namespace, out_dir: Path, obs0: np.ndarray) -> list[dict[str, Any]]:
    print("[suite] === EXP 1: many within-window seeds (peak distribution) ===", flush=True)
    many = args.many_seeds.strip() or "0-19"
    seeds = _parse_seed_list(many)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        sh = within_window_day_shuffle(obs0, seed=seed)
        print(f"[suite] exp1 ww-shuffle seed={seed} …", flush=True)
        rec = _fit(
            sh,
            tune=args.tune_many,
            draws=args.draws_many,
            chains=args.chains_many,
            nuts_backend=args.nuts_backend,
        )
        rec.update({"exp": 1, "arm": "within_window_day_shuffle", "seed": seed})
        rows.append(rec)
        print(
            f"[suite] exp1 seed={seed} E[τ]={rec['e_tau']:.3f} peak={rec['peak_tau']} "
            f"p_peak={rec['peak_p']:.3f} MAD_u={rec['mean_abs_dev_from_uniform']:.4f}",
            flush=True,
        )
    peaks = [r["peak_tau"] for r in rows]
    e_taus = [r["e_tau"] for r in rows]
    mads = [r["mean_abs_dev_from_uniform"] for r in rows]
    stack = np.vstack([r["mean_p_tau"] for r in rows])
    summary = {
        "exp": 1,
        "arm": "summary",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "peak_counts": {str(k): int(peaks.count(k)) for k in sorted(set(peaks))},
        "e_tau_mean": float(np.mean(e_taus)),
        "e_tau_std": float(np.std(e_taus, ddof=0)),
        "e_tau_min": float(np.min(e_taus)),
        "e_tau_max": float(np.max(e_taus)),
        "mad_mean": float(np.mean(mads)),
        "mad_std": float(np.std(mads, ddof=0)),
        "frac_peak_tau7": float(peaks.count(7) / max(1, len(peaks))),
        "frac_mad_below_0_05": float(sum(m < 0.05 for m in mads) / max(1, len(mads))),
        "tau_support": rows[0]["tau_support"],
        "mean_p_tau": [float(x) for x in stack.mean(axis=0)],
        "mean_abs_dev_from_uniform": _pmf_mad_uniform(stack.mean(axis=0)),
    }
    rows.append(summary)
    _save_results(out_dir / "exp1_many_seeds.json", rows)
    print(f"[suite] exp1 summary peak_counts={summary['peak_counts']}", flush=True)
    return rows


def run_exp2(args: argparse.Namespace, out_dir: Path, obs0: np.ndarray) -> list[dict[str, Any]]:
    print("[suite] === EXP 2: column (day-index) shuffle ===", flush=True)
    rows: list[dict[str, Any]] = []
    zarr = args.real_dir / "trace.zarr"
    if zarr.exists():
        import arviz as az

        idata = az.from_zarr(str(zarr))
        support, probs = tau_probabilities(idata)
        support_i = [int(x) for x in np.asarray(support).ravel()]
        probs_f = [float(x) for x in np.asarray(probs, dtype=float).ravel()]
        rows.append(
            {
                "exp": 2,
                "arm": "real_stored",
                "seed": None,
                "e_tau": float(np.asarray(idata.posterior["tau_mean"]).mean()),
                "tau_support": support_i,
                "mean_p_tau": probs_f,
                "peak_tau": _peak_k(support_i, probs_f),
                "peak_p": float(max(probs_f)),
                "mean_abs_dev_from_uniform": _pmf_mad_uniform(probs_f),
                "source": "stored_trace.zarr",
            }
        )
    seeds = _parse_seed_list(args.seeds)
    for seed in seeds:
        col = column_day_shuffle(obs0, seed=seed)
        np.savez_compressed(
            out_dir / f"exp2_observations_column_seed{seed}.npz", daily__mean=col
        )
        print(f"[suite] exp2 column-shuffle seed={seed} …", flush=True)
        rec = _fit(
            col,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            nuts_backend=args.nuts_backend,
        )
        rec.update({"exp": 2, "arm": "column_day_shuffle", "seed": seed})
        rows.append(rec)
        print(
            f"[suite] exp2 seed={seed} E[τ]={rec['e_tau']:.3f} peak={rec['peak_tau']} "
            f"MAD_u={rec['mean_abs_dev_from_uniform']:.4f}",
            flush=True,
        )
    _save_results(out_dir / "exp2_column_shuffle.json", rows)
    return rows


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exps = [int(x) for x in str(args.exps).split(",") if str(x).strip()]
    obs0 = _load_observations(args.real_dir)
    np.savez_compressed(args.out_dir / "observations_real.npz", daily__mean=obs0)
    meta = {
        "fingerprint": FINGERPRINT,
        "exps": exps,
        "tune": args.tune,
        "draws": args.draws,
        "chains": args.chains,
        "tune_many": args.tune_many,
        "draws_many": args.draws_many,
        "chains_many": args.chains_many,
        "seeds": _parse_seed_list(args.seeds),
        "many_seeds": _parse_seed_list(args.many_seeds.strip() or "0-19"),
        "synth_tau": args.synth_tau,
        "observations_shape": list(obs0.shape),
    }
    (args.out_dir / "suite_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    all_rows: list[dict[str, Any]] = []
    if 4 in exps:
        all_rows.extend(run_exp4(args, args.out_dir))
    if 3 in exps:
        all_rows.extend(run_exp3(args, args.out_dir, obs0))
    if 1 in exps:
        all_rows.extend(run_exp1(args, args.out_dir, obs0))
    if 2 in exps:
        all_rows.extend(run_exp2(args, args.out_dir, obs0))

    _save_results(args.out_dir / "suite_all_results.json", all_rows)
    print("[suite] done", flush=True)


if __name__ == "__main__":
    main()
