#!/usr/bin/env python3
"""Diagnose Beta boundary blow-up for daily:range and optionally refit constrained priors.

Reads exhaustive_search_parallel.csv + profile caches / top10_refits posterior summaries.
Optionally runs a narrow smoke MCMC subset:
  daily range | mean+range × N∈{24,48} × ov∈{0,0.25,0.5}
  × {beta, beta_constrained, interval_inflated_beta} (+ student_t/normal for mean).

Usage (from seismic_pipeline_standalone/):
  python scripts/diagnose_beta_boundary.py
  python scripts/diagnose_beta_boundary.py --refit-smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import (
    _beta_pdf_unit,
    _values_flat,
    score_changepoint_trace,
)
from seismic_pipeline.bayesian.search_export import elpd_column_legend_markdown
from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
from seismic_pipeline.features.runtime import set_runtime_export_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "run_output_8day_parallel_full" / "exhaustive_search_parallel.csv",
    )
    p.add_argument(
        "--profile-cache",
        type=Path,
        default=PROJECT_ROOT / "run_output_8day_parallel_full" / "profile_cache",
    )
    p.add_argument(
        "--refits-dir",
        type=Path,
        default=PROJECT_ROOT / "run_output_8day_parallel_full" / "top10_refits",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "run_output_8day_parallel_full" / "beta_boundary_diag",
    )
    p.add_argument("--support-upper", type=float, default=2.0)
    p.add_argument("--near-hi", type=float, default=0.9)
    p.add_argument("--near-lo", type=float, default=0.1)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--refit-smoke", action="store_true", help="Run narrow MCMC smoke subset.")
    p.add_argument("--tune", type=int, default=400)
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument(
        "--smoke-max",
        type=int,
        default=6,
        help="Max smoke configs to refit (keep small).",
    )
    return p.parse_args()


def _profile_csv(cache_root: Path, n_points: int, overlap: float) -> Path:
    name = f"rem_n{int(n_points)}_ov{float(overlap):.2f}_stage2"
    return cache_root / name / "samples_10days_nanpad.csv"


def _range_boundary_mass(
    csv_path: Path,
    *,
    n_points: int,
    support_upper: float,
    near_hi: float,
    near_lo: float,
    eps: float,
) -> dict[str, Any]:
    set_runtime_export_cfg(
        {
            "n_points_per_day": int(n_points),
            "overlap": 0.0,  # overwritten by caller via path; features use n_points
            "rem_stage": 2,
        }
    )
    prep = prepare_model_data(str(csv_path))
    group_data = build_group_data(
        prep["data_norm"],
        n_chunks=8,
        feature_selection={"daily": ["range"]},
        data_raw=prep["data_raw"],
        n_points_per_day=int(n_points),
        fixed_n_days=8,
    )
    y = np.asarray(group_data["daily"]["range"].to_numpy(), dtype=float).ravel()
    y = y[np.isfinite(y)]
    y_unit = np.clip(y / float(support_upper), eps, 1.0 - eps)
    n = int(y_unit.size)
    frac_hi = float(np.mean(y_unit >= near_hi)) if n else float("nan")
    frac_lo = float(np.mean(y_unit <= near_lo)) if n else float("nan")
    return {
        "n_obs": n,
        "y_min": float(np.min(y)) if n else float("nan"),
        "y_max": float(np.max(y)) if n else float("nan"),
        "y_unit_mean": float(np.mean(y_unit)) if n else float("nan"),
        "frac_ge_0.9": frac_hi,
        "frac_le_0.1": frac_lo,
        "frac_clipped_hi": float(np.mean(y / support_upper >= 1.0 - eps)) if n else float("nan"),
        "support_upper": float(support_upper),
    }


def _posterior_shape_lt1_from_summary(summary_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(summary_csv)
    name_col = df.columns[0]
    out: dict[str, Any] = {"path": str(summary_csv), "params": {}}
    for _, row in df.iterrows():
        name = str(row[name_col])
        if not (name.startswith("alpha_") or name.startswith("beta_")):
            continue
        mean = float(row["mean"])
        sd = float(row["sd"])
        # Normal approx for P(param < 1) from summary mean/sd.
        from math import erf, sqrt

        z = (1.0 - mean) / max(sd, 1e-12)
        # Φ(z) ≈ 0.5 (1+erf(z/√2))
        p_lt1 = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        out["params"][name] = {
            "mean": mean,
            "sd": sd,
            "p_lt1_normal_approx": float(p_lt1),
            "mean_lt1": bool(mean < 1.0),
        }
    shape_means = [v["mean"] for k, v in out["params"].items() if k.startswith("beta_")]
    out["any_beta_mean_lt1"] = any(m < 1.0 for m in shape_means)
    out["n_beta_mean_lt1"] = int(sum(1 for m in shape_means if m < 1.0))
    return out


def _ll_vs_y_correlation(
    y_unit: np.ndarray,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    y = np.asarray(y_unit, dtype=float).ravel()
    y = y[np.isfinite(y)]
    ll = np.log(_beta_pdf_unit(y, alpha=alpha, beta=beta) + 1e-300)
    if y.size < 3:
        return {"corr_y_ll": float("nan"), "n": float(y.size)}
    corr = float(np.corrcoef(y, ll)[0, 1])
    return {
        "corr_y_ll": corr,
        "n": float(y.size),
        "mean_ll": float(np.mean(ll)),
        "ll_at_0.5": float(np.log(_beta_pdf_unit(np.array([0.5]), alpha, beta)[0])),
        "ll_at_0.95": float(np.log(_beta_pdf_unit(np.array([0.95]), alpha, beta)[0])),
    }


def _csv_leadership(df: pd.DataFrame) -> dict[str, Any]:
    elig = df.copy()
    if "rank_eligible" in elig.columns:
        elig = elig[elig["rank_eligible"].fillna(False).astype(bool)]
    sort_col = "elpd_loo_per_feature_event"
    top = elig.sort_values(sort_col, ascending=False).head(15)
    cols = [
        c
        for c in [
            "rank_by_loo",
            "features",
            "likelihoods",
            "n_points",
            "overlap",
            "elpd_loo",
            "elpd_loo_per_feature_event",
            "loo_ic",
            "e_tau",
        ]
        if c in top.columns
    ]
    # Plain beta vs IIB for daily:range only
    mask_range = top["features"].astype(str).str.contains(r"daily:\s*range", regex=True) & ~top[
        "features"
    ].astype(str).str.contains("mean")
    comparisons = []
    for n, ov in [(48, 0.0), (48, 0.25), (48, 0.5), (24, 0.0)]:
        sub = df[
            (df["features"].astype(str).str.match(r"^\s*daily:\s*range\s*$"))
            & (df["n_points"] == n)
            & (np.isclose(df["overlap"].astype(float), ov))
        ]
        row: dict[str, Any] = {"n_points": n, "overlap": ov}
        for _, r in sub.iterrows():
            lik = str(r["likelihoods"])
            row[lik] = {
                "elpd_loo_per_feature_event": float(r[sort_col]),
                "elpd_loo": float(r["elpd_loo"]),
                "e_tau": float(r["e_tau"]),
            }
        comparisons.append(row)
    return {
        "top15": top[cols].to_dict(orient="records"),
        "daily_range_beta_vs_iib": comparisons,
        "loo_ic_is_minus_2_elpd": bool(
            np.isclose(
                pd.to_numeric(df["loo_ic"], errors="coerce"),
                -2.0 * pd.to_numeric(df["elpd_loo"], errors="coerce"),
                equal_nan=True,
            ).mean()
            > 0.999
        ),
    }


def _refit_one(
    *,
    csv_path: Path,
    n_points: int,
    overlap: float,
    feature_selection: dict[str, list[str]],
    parameter_selection: dict[str, dict[str, Any]],
    tune: int,
    draws: int,
    chains: int,
) -> dict[str, Any]:
    set_runtime_export_cfg(
        {
            "n_points_per_day": int(n_points),
            "overlap": float(overlap),
            "rem_stage": 2,
        }
    )
    prep = prepare_model_data(str(csv_path))
    group_data = build_group_data(
        prep["data_norm"],
        n_chunks=8,
        feature_selection=feature_selection,
        data_raw=prep["data_raw"],
        n_points_per_day=int(n_points),
        fixed_n_days=8,
    )
    model = build_changepoint_model(
        group_data,
        tau_lower=3,
        tau_upper=8,
        parameter_selection=parameter_selection,
        tau_mode="marginalized",
    )
    trace = sample_model(
        model,
        draws=int(draws),
        tune=int(tune),
        chains=int(chains),
        cores=1,
        progressbar=False,
        nuts_backend="blackjax",
    )
    score = score_changepoint_trace(
        trace,
        group_data=group_data,
        parameter_selection=parameter_selection,
        model=model,
        criterion="loo",
        warn_on_fallback=False,
        loo_report="elpd",
    )
    n_feat = sum(len(v) for v in feature_selection.values())
    n_events = int(next(iter(next(iter(group_data.values())).values())).shape[0])
    elpd = float(score.get("elpd_loo", float("nan")))
    shape_stats: dict[str, Any] = {}
    from seismic_pipeline.bayesian.diagnostics import _available_varnames

    for varname in _available_varnames(trace):
        if varname.startswith("alpha_") or varname.startswith("beta_"):
            if varname.endswith("_raw"):
                continue
            vals = _values_flat(trace, varname)
            shape_stats[varname] = {
                "mean": float(np.mean(vals)),
                "frac_lt1": float(np.mean(vals < 1.0)),
            }
    return {
        "features": feature_selection,
        "likelihoods": {k: v.get("likelihood") for k, v in parameter_selection.items()},
        "n_points": int(n_points),
        "overlap": float(overlap),
        "elpd_loo": elpd,
        "loo_ic": float(score.get("loo_ic", float("nan"))),
        "elpd_loo_per_feature_event": elpd / float(n_feat * n_events) if n_feat and n_events else float("nan"),
        "e_tau": float(score.get("e_tau", float("nan"))),
        "r_hat_max": float(score.get("r_hat_max", float("nan"))),
        "n_divergences": int(score.get("n_divergences", 0)),
        "shape_stats": shape_stats,
        "n_events": n_events,
        "n_features": n_feat,
    }


def _smoke_configs() -> list[dict[str, Any]]:
    """Manageable subset: daily range / mean+range at key grids × likelihoods."""
    configs: list[dict[str, Any]] = []
    for n, ov in [(48, 0.0), (24, 0.0)]:
        for lik in ("beta", "beta_constrained", "interval_inflated_beta"):
            entry: dict[str, Any] = {"likelihood": lik, "support_upper": 2.0}
            if lik == "interval_inflated_beta":
                entry["threshold"] = 0.9
            configs.append(
                {
                    "feature_selection": {"daily": ["range"]},
                    "parameter_selection": {"range": entry},
                    "n_points": n,
                    "overlap": ov,
                }
            )
        for mean_lik in ("student_t", "normal"):
            for range_lik in ("beta_constrained", "interval_inflated_beta"):
                r_entry: dict[str, Any] = {"likelihood": range_lik, "support_upper": 2.0}
                if range_lik == "interval_inflated_beta":
                    r_entry["threshold"] = 0.9
                configs.append(
                    {
                        "feature_selection": {"daily": ["mean", "range"]},
                        "parameter_selection": {
                            "mean": {"likelihood": mean_lik},
                            "range": r_entry,
                        },
                        "n_points": n,
                        "overlap": ov,
                    }
                )
    return configs


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    leadership = _csv_leadership(df)

    # Boundary mass for top grids
    boundary_rows = []
    for n, ov in [(48, 0.0), (48, 0.25), (48, 0.5), (24, 0.0), (24, 0.25), (24, 0.5)]:
        csv_path = _profile_csv(args.profile_cache, n, ov)
        if not csv_path.is_file():
            boundary_rows.append({"n_points": n, "overlap": ov, "error": f"missing {csv_path}"})
            continue
        # Update export cfg with correct overlap for documentation; build uses n_points.
        set_runtime_export_cfg(
            {"n_points_per_day": int(n), "overlap": float(ov), "rem_stage": 2}
        )
        stats = _range_boundary_mass(
            csv_path,
            n_points=n,
            support_upper=args.support_upper,
            near_hi=args.near_hi,
            near_lo=args.near_lo,
            eps=args.eps,
        )
        stats["n_points"] = n
        stats["overlap"] = ov
        boundary_rows.append(stats)

    # Posterior α/β < 1 from top10 refits
    posterior_rows = []
    ll_corr_rows = []
    if args.refits_dir.is_dir():
        for rank_dir in sorted(args.refits_dir.glob("rank*")):
            summary = rank_dir / "posterior_summary.csv"
            meta_path = rank_dir / "refit_meta.json"
            if not summary.is_file():
                continue
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            shape = _posterior_shape_lt1_from_summary(summary)
            shape["fingerprint"] = meta.get("fingerprint")
            shape["features"] = meta.get("features")
            shape["likelihoods"] = meta.get("likelihoods")
            shape["rank_by_loo"] = meta.get("rank_by_loo")
            posterior_rows.append(shape)

            # LL vs y for range-beta refits using posterior means
            lik = str(meta.get("likelihoods", ""))
            if "range=beta" in lik and "interval" not in lik:
                rem = (meta.get("config") or {}).get("rem_profile_params") or {}
                n = int(rem.get("n_points_per_day", meta.get("n_points", 48)))
                ov = float(rem.get("overlap", meta.get("overlap", 0.0)))
                csv_path = _profile_csv(args.profile_cache, n, ov)
                if csv_path.is_file():
                    set_runtime_export_cfg(
                        {"n_points_per_day": n, "overlap": ov, "rem_stage": 2}
                    )
                    prep = prepare_model_data(str(csv_path))
                    gd = build_group_data(
                        prep["data_norm"],
                        n_chunks=8,
                        feature_selection={"daily": ["range"]},
                        data_raw=prep["data_raw"],
                        n_points_per_day=n,
                        fixed_n_days=8,
                    )
                    y = np.asarray(gd["daily"]["range"].to_numpy(), dtype=float).ravel()
                    y_unit = np.clip(y / args.support_upper, args.eps, 1.0 - args.eps)
                    params = shape["params"]
                    a1 = params.get("alpha_daily_range_1", {}).get("mean")
                    b1 = params.get("beta_daily_range_1", {}).get("mean")
                    if a1 is not None and b1 is not None:
                        corr = _ll_vs_y_correlation(y_unit, float(a1), float(b1))
                        corr["rank"] = meta.get("rank_by_loo")
                        corr["alpha"] = float(a1)
                        corr["beta"] = float(b1)
                        corr["regime"] = 1
                        ll_corr_rows.append(corr)

    smoke_results: list[dict[str, Any]] = []
    if args.refit_smoke:
        for i, cfg in enumerate(_smoke_configs()):
            if i >= int(args.smoke_max):
                break
            n = int(cfg["n_points"])
            ov = float(cfg["overlap"])
            csv_path = _profile_csv(args.profile_cache, n, ov)
            if not csv_path.is_file():
                smoke_results.append({"error": f"missing {csv_path}", **cfg})
                continue
            print(
                f"[smoke {i+1}/{args.smoke_max}] N={n} ov={ov} "
                f"feats={cfg['feature_selection']} lik={cfg['parameter_selection']}",
                flush=True,
            )
            try:
                smoke_results.append(
                    _refit_one(
                        csv_path=csv_path,
                        n_points=n,
                        overlap=ov,
                        feature_selection=cfg["feature_selection"],
                        parameter_selection=cfg["parameter_selection"],
                        tune=args.tune,
                        draws=args.draws,
                        chains=args.chains,
                    )
                )
            except Exception as exc:
                smoke_results.append({"error": str(exc), **cfg})

    payload = {
        "csv": str(args.csv),
        "leadership": leadership,
        "boundary_mass": boundary_rows,
        "posterior_shapes_from_refits": posterior_rows,
        "ll_vs_y": ll_corr_rows,
        "smoke_refits": smoke_results,
        "elpd_legend_md": elpd_column_legend_markdown(),
    }
    out_json = args.out_dir / "beta_boundary_diagnostics.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)

    # Quick stdout summary
    print("\n=== Boundary mass (range/support_upper) ===", flush=True)
    for row in boundary_rows:
        if "error" in row:
            print(f"  N={row.get('n_points')} ov={row.get('overlap')}: {row['error']}")
            continue
        print(
            f"  N={row['n_points']} ov={row['overlap']}: "
            f"frac≥0.9={row['frac_ge_0.9']:.3f} frac≤0.1={row['frac_le_0.1']:.3f} "
            f"max_raw={row['y_max']:.3f}"
        )
    print("\n=== Posterior β means < 1 (top refits) ===", flush=True)
    for row in posterior_rows:
        if not row.get("params"):
            continue
        betas = {k: v for k, v in row["params"].items() if k.startswith("beta_")}
        if not betas:
            continue
        print(
            f"  rank={row.get('rank_by_loo')} {row.get('likelihoods')}: "
            + ", ".join(f"{k}={v['mean']:.3f}" for k, v in betas.items())
        )
    if ll_corr_rows:
        print("\n=== corr(y, log-lik) under posterior mean Beta ===", flush=True)
        for row in ll_corr_rows:
            print(
                f"  rank={row.get('rank')} α={row['alpha']:.2f} β={row['beta']:.2f} "
                f"corr={row['corr_y_ll']:.3f} ll@0.95={row['ll_at_0.95']:.2f} "
                f"ll@0.5={row['ll_at_0.5']:.2f}"
            )
    if smoke_results:
        print("\n=== Smoke refits ===", flush=True)
        for row in smoke_results:
            if "error" in row:
                print(f"  ERROR: {row['error']}")
                continue
            print(
                f"  N={row['n_points']} ov={row['overlap']} lik={row['likelihoods']} "
                f"elpd/feat·evt={row['elpd_loo_per_feature_event']:.3f} "
                f"E[τ]={row['e_tau']:.2f} "
                f"β_frac<1="
                + str(
                    {
                        k: round(v["frac_lt1"], 3)
                        for k, v in row.get("shape_stats", {}).items()
                        if k.startswith("beta_")
                    }
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
