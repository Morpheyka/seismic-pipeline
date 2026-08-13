#!/usr/bin/env python3
"""Within-window day-shuffle null for primary mean-only cell ``4f6e4c855d72864d``.

For each event window, permute finite day values among valid day positions and
keep NaN masks fixed. Fit the same student_t daily:mean model as the real
refit; dump mean p(τ=k) for real + each shuffle seed.

Usage (from repo root or standalone/):
  ../.venv/bin/python scripts/run_within_window_day_shuffle.py
  ../.venv/bin/python scripts/run_within_window_day_shuffle.py --tune 6000 --draws 3000 --chains 4
  ../.venv/bin/python scripts/run_within_window_day_shuffle.py --skip-real-refit  # use stored tau_probs
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
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "run_output_8day_density_safe" / "within_window_shuffle"
)

FEATURE_SELECTION = {"daily": ["mean"]}
PARAMETER_SELECTION = {"mean": {"likelihood": "student_t"}}
TAU_LOWER = 2
TAU_UPPER = 8
FINGERPRINT = "4f6e4c855d72864d"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tune", type=int, default=6000)
    p.add_argument("--draws", type=int, default=3000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--nuts-backend", default="blackjax")
    p.add_argument(
        "--skip-real-refit",
        action="store_true",
        help="Load mean p(τ=k) from real-dir/trace.zarr instead of re-fitting real.",
    )
    p.add_argument(
        "--refit-real",
        action="store_true",
        help="Force MCMC on unshuffled observations (same budget as shuffle).",
    )
    return p.parse_args()


def _load_observations(real_dir: Path) -> np.ndarray:
    path = real_dir / "observations.npz"
    if not path.is_file():
        raise SystemExit(f"Missing observations: {path}")
    z = np.load(path)
    if "daily__mean" not in z:
        raise SystemExit(f"observations.npz missing daily__mean; keys={list(z.keys())}")
    return np.asarray(z["daily__mean"], dtype=float)


def within_window_day_shuffle(obs: np.ndarray, *, seed: int) -> tuple[np.ndarray, int]:
    """Permute finite day values inside each window; keep NaN positions fixed."""
    out = np.array(obs, dtype=float, copy=True)
    rng = np.random.default_rng(int(seed))
    n_windows, _n_days = out.shape
    n_moved = 0
    for i in range(n_windows):
        valid = np.isfinite(out[i])
        idxs = np.flatnonzero(valid)
        if idxs.size < 2:
            continue
        values = out[i, idxs].copy()
        perm = rng.permutation(idxs.size)
        out[i, idxs] = values[perm]
        if not np.array_equal(perm, np.arange(idxs.size)):
            n_moved += 1
    return out, n_moved


def _group_data_from_obs(obs: np.ndarray) -> dict[str, dict[str, pd.DataFrame]]:
    return {"daily": {"mean": pd.DataFrame(np.asarray(obs, dtype=float))}}


def _mean_pmf_from_trace(trace) -> tuple[list[int], list[float]]:
    support, probs = tau_probabilities(trace)
    support_i = [int(x) for x in np.asarray(support).ravel()]
    probs_f = [float(x) for x in np.asarray(probs, dtype=float).ravel()]
    return support_i, probs_f


def _mean_pmf_from_zarr(trace_path: Path) -> tuple[list[int], list[float], float]:
    import arviz as az

    idata = az.from_zarr(str(trace_path))
    support, probs = tau_probabilities(idata)
    e_tau = float(np.asarray(idata.posterior["tau_mean"]).mean())
    return (
        [int(x) for x in np.asarray(support).ravel()],
        [float(x) for x in np.asarray(probs, dtype=float).ravel()],
        e_tau,
    )


def _fit(
    obs: np.ndarray,
    *,
    tune: int,
    draws: int,
    chains: int,
    nuts_backend: str,
) -> tuple[Any, dict[str, Any]]:
    group_data = _group_data_from_obs(obs)
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
    support, probs = _mean_pmf_from_trace(trace)
    n_feat = 1
    n_events = int(obs.shape[0])
    elpd = float(scores.get("elpd_loo", float("nan")))
    rec: dict[str, Any] = {
        "e_tau": float(scores.get("e_tau", float("nan"))),
        "tau_hdi_60_width": float(scores.get("tau_hdi_60_width", float("nan"))),
        "elpd_loo": elpd,
        "elpd_loo_per_feature_event": elpd / max(1, n_feat * n_events),
        "n_events": n_events,
        "r_hat_max": float(scores.get("r_hat_max", float("nan"))),
        "ess_min_bulk": float(scores.get("ess_min_bulk", float("nan"))),
        "tau_support": support,
        "mean_p_tau": probs,
        "mean_p_tau_by_k": {str(k): p for k, p in zip(support, probs)},
        "uniform_p": 1.0 / float(len(support)) if support else float("nan"),
        "sample_s": float(sample_s),
    }
    return trace, rec


def _pmf_l1_to_uniform(probs: list[float]) -> float:
    if not probs:
        return float("nan")
    u = 1.0 / float(len(probs))
    return float(np.mean(np.abs(np.asarray(probs, dtype=float) - u)))


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs0 = _load_observations(args.real_dir)
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]

    meta = {
        "fingerprint": FINGERPRINT,
        "null": "within_window_day_shuffle",
        "note": (
            "Permute finite day values inside each window; NaN day masks stay fixed. "
            "Same model as primary mean-only cell."
        ),
        "feature_selection": FEATURE_SELECTION,
        "parameter_selection": PARAMETER_SELECTION,
        "tau_lower": TAU_LOWER,
        "tau_upper": TAU_UPPER,
        "n_points": 24,
        "overlap": 0.0,
        "tune": args.tune,
        "draws": args.draws,
        "chains": args.chains,
        "nuts_backend": args.nuts_backend,
        "seeds": seeds,
        "real_dir": str(args.real_dir),
        "observations_shape": list(obs0.shape),
        "n_nan_days": int(np.isnan(obs0).sum()),
    }
    (args.out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.out_dir / "observations_real.npz", daily__mean=obs0)

    results: list[dict[str, Any]] = []
    use_stored_real = bool(args.skip_real_refit) and not bool(args.refit_real)
    if use_stored_real:
        trace_path = args.real_dir / "trace.zarr"
        print(f"[ww-shuffle] loading real tau_probs from {trace_path}", flush=True)
        support, probs, e_tau = _mean_pmf_from_zarr(trace_path)
        real_rec: dict[str, Any] = {
            "arm": "real",
            "seed": None,
            "source": "stored_trace.zarr",
            "e_tau": e_tau,
            "n_events": int(obs0.shape[0]),
            "tau_support": support,
            "mean_p_tau": probs,
            "mean_p_tau_by_k": {str(k): p for k, p in zip(support, probs)},
            "uniform_p": 1.0 / float(len(support)),
            "mean_abs_dev_from_uniform": _pmf_l1_to_uniform(probs),
            "tune": None,
            "draws": None,
            "chains": None,
            "note": "Real comparator from confirmatory refit artefacts (not re-sampled).",
        }
        # Fill MCMC budget fields from refit_meta if present
        meta_path = args.real_dir / "refit_meta.json"
        if meta_path.is_file():
            rm = json.loads(meta_path.read_text(encoding="utf-8"))
            real_rec["tune"] = rm.get("tune")
            real_rec["draws"] = rm.get("draws")
            real_rec["chains"] = rm.get("chains")
        results.append(real_rec)
        print(
            f"[ww-shuffle] real E[τ]={e_tau:.3f} mean_p={np.round(probs, 3)} "
            f"MAD_uniform={real_rec['mean_abs_dev_from_uniform']:.4f}",
            flush=True,
        )
    else:
        print("[ww-shuffle] fitting real (unshuffled) …", flush=True)
        _, real_rec = _fit(
            obs0,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            nuts_backend=args.nuts_backend,
        )
        real_rec.update(
            {
                "arm": "real",
                "seed": None,
                "source": "mcmc_this_run",
                "mean_abs_dev_from_uniform": _pmf_l1_to_uniform(real_rec["mean_p_tau"]),
                "tune": args.tune,
                "draws": args.draws,
                "chains": args.chains,
            }
        )
        results.append(real_rec)
        print(
            f"[ww-shuffle] real E[τ]={real_rec['e_tau']:.3f} "
            f"mean_p={np.round(real_rec['mean_p_tau'], 3)} "
            f"sample_s={real_rec['sample_s']:.1f}",
            flush=True,
        )

    for seed in seeds:
        print(f"[ww-shuffle] seed={seed} within-window permute …", flush=True)
        shuffled, n_windows_moved = within_window_day_shuffle(obs0, seed=seed)
        np.savez_compressed(
            args.out_dir / f"observations_shuffle_seed{seed}.npz",
            daily__mean=shuffled,
        )
        _, rec = _fit(
            shuffled,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            nuts_backend=args.nuts_backend,
        )
        rec.update(
            {
                "arm": "within_window_day_shuffle",
                "seed": seed,
                "source": "mcmc_this_run",
                "n_windows_with_permutation": int(n_windows_moved),
                "mean_abs_dev_from_uniform": _pmf_l1_to_uniform(rec["mean_p_tau"]),
                "tune": args.tune,
                "draws": args.draws,
                "chains": args.chains,
            }
        )
        results.append(rec)
        print(
            f"[ww-shuffle] seed={seed} E[τ]={rec['e_tau']:.3f} "
            f"mean_p={np.round(rec['mean_p_tau'], 3)} "
            f"MAD_uniform={rec['mean_abs_dev_from_uniform']:.4f} "
            f"sample_s={rec['sample_s']:.1f}",
            flush=True,
        )

    # Aggregate shuffle mean PMF across seeds
    shuffle_rows = [r for r in results if r["arm"] == "within_window_day_shuffle"]
    if shuffle_rows:
        support = shuffle_rows[0]["tau_support"]
        stack = np.vstack([np.asarray(r["mean_p_tau"], dtype=float) for r in shuffle_rows])
        agg = {
            "arm": "shuffle_seeds_mean",
            "seeds": seeds,
            "tau_support": support,
            "mean_p_tau": [float(x) for x in stack.mean(axis=0)],
            "std_p_tau": [float(x) for x in stack.std(axis=0, ddof=0)],
            "mean_p_tau_by_k": {
                str(k): float(v) for k, v in zip(support, stack.mean(axis=0))
            },
            "uniform_p": 1.0 / float(len(support)),
            "mean_abs_dev_from_uniform": _pmf_l1_to_uniform(
                [float(x) for x in stack.mean(axis=0)]
            ),
            "e_tau_mean": float(np.mean([r["e_tau"] for r in shuffle_rows])),
            "e_tau_std": float(np.std([r["e_tau"] for r in shuffle_rows], ddof=0)),
        }
        results.append(agg)

    out_json = args.out_dir / "within_window_shuffle_results.json"
    out_csv = args.out_dir / "within_window_shuffle_results.csv"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Flatten PMF columns for CSV
    flat_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in {"mean_p_tau_by_k", "tau_support", "mean_p_tau", "std_p_tau"}}
        support = r.get("tau_support") or []
        probs = r.get("mean_p_tau") or []
        for k, p in zip(support, probs):
            row[f"p_tau_{k}"] = p
        if "std_p_tau" in r:
            for k, s in zip(support, r["std_p_tau"]):
                row[f"std_p_tau_{k}"] = s
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"[ww-shuffle] wrote {out_json}", flush=True)
    print(f"[ww-shuffle] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
