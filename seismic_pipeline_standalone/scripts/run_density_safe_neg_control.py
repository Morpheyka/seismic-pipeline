#!/usr/bin/env python3
"""Negative control on Primary A with day-mask: shuffle_dates seeds 0,1,2.

Uses row-shuffle-within-rat proxy on density-safe masked cohort when calendar
re-export for shuffled dates is unavailable (same integrity idea).

Usage:
  python scripts/run_density_safe_neg_control.py
  python scripts/run_density_safe_neg_control.py --seeds 0,1,2 --tune 400 --draws 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import score_changepoint_trace
from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
from seismic_pipeline.features.runtime import set_runtime_export_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile-csv",
        type=Path,
        default=(
            PROJECT_ROOT
            / "run_output_8day_density_safe"
            / "profile_cache"
            / "rem_n24_ov0.50_stage2"
            / "samples_10days_nanpad.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "run_output_density_safe_neg_control",
    )
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tune", type=int, default=400)
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--nuts-backend", default="blackjax")
    return p.parse_args()


def _fit(
    group_data: dict,
    *,
    tune: int,
    draws: int,
    chains: int,
    nuts_backend: str,
) -> dict[str, Any]:
    ps = {
        "mean": {"likelihood": "student_t"},
        "range": {"likelihood": "beta_constrained", "support_upper": 2.0},
    }
    fs = {"daily": ["mean", "range"]}
    model = build_changepoint_model(
        group_data,
        tau_lower=2,
        tau_upper=8,
        parameter_selection=ps,
        tau_mode="marginalized",
    )
    kwargs: dict[str, Any] = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        progressbar=False,
        nuts_backend=nuts_backend,
    )
    if nuts_backend == "pymc":
        kwargs["cores"] = 1
    else:
        kwargs["jax_chain_method"] = "parallel"
        kwargs["jax_var_names"] = [
            "changepoint_pointwise_log_lik",
            "tau_probs",
            "tau_support",
            "tau_mean",
        ]
        kwargs["materialize_posterior_vars"] = kwargs["jax_var_names"]
    trace = sample_model(model, **kwargs)
    scores = score_changepoint_trace(
        trace,
        group_data=group_data,
        parameter_selection=ps,
        model=model,
        criterion="loo",
        warn_on_fallback=False,
        loo_report="elpd",
    )
    n_feat = 2
    n_events = int(next(iter(next(iter(group_data.values())).values())).shape[0])
    elpd = float(scores.get("elpd_loo", float("nan")))
    return {
        "e_tau": float(scores.get("e_tau", float("nan"))),
        "tau_hdi_60_width": float(scores.get("tau_hdi_60_width", float("nan"))),
        "elpd_loo": elpd,
        "elpd_loo_per_feature_event": elpd / max(1, n_feat * n_events),
        "n_events": n_events,
        "r_hat_max": float(scores.get("r_hat_max", float("nan"))),
        "ess_min_bulk": float(scores.get("ess_min_bulk", float("nan"))),
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.profile_csv
    if not csv_path.is_file():
        raise SystemExit(f"Missing profile CSV: {csv_path}")

    set_runtime_export_cfg({"n_points_per_day": 24, "overlap": 0.5, "rem_stage": 2})
    prep = prepare_model_data(
        str(csv_path),
        day_mask=True,
        apply_artifacts=True,
        min_valid_days=6,
        n_points_per_day=24,
        window_days=8,
    )
    meta_path = csv_path.with_name(csv_path.name.replace("_nanpad.csv", "_metadata.csv"))
    meta = pd.read_csv(meta_path)
    exported = meta[meta["exported"].astype(bool)].sort_values("row_index").reset_index(drop=True)
    # Align meta to good_indices (cohort after K=6).
    good = np.asarray(prep["good_indices"], dtype=int)
    exported = exported.iloc[good].reset_index(drop=True)

    data_norm0 = np.asarray(prep["data_norm"], dtype=float)
    data_raw0 = np.asarray(prep["data_raw"], dtype=float)
    day_valid0 = prep.get("day_valid")

    # Real-date baseline
    print("[neg] fitting real-date Primary A …", flush=True)
    gd_real = build_group_data(
        data_norm0,
        n_chunks=8,
        feature_selection={"daily": ["mean", "range"]},
        data_raw=data_raw0,
        n_points_per_day=24,
        fixed_n_days=8,
        day_valid=day_valid0,
    )
    real = _fit(
        gd_real,
        tune=args.tune,
        draws=args.draws,
        chains=args.chains,
        nuts_backend=args.nuts_backend,
    )
    real["arm"] = "real_dates"
    real["seed"] = None
    print(f"[neg] real E[τ]={real['e_tau']:.3f} elpd/fe={real['elpd_loo_per_feature_event']:.3f}")

    results = [real]
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    for seed in seeds:
        print(f"[neg] seed={seed} row-shuffle within rat …", flush=True)
        data_norm = data_norm0.copy()
        data_raw = data_raw0.copy()
        day_valid = None if day_valid0 is None else np.asarray(day_valid0, dtype=bool).copy()
        rng = random.Random(seed)
        by_rat: dict[str, list[int]] = {}
        for i, row in exported.iterrows():
            by_rat.setdefault(str(row["rat_id"]), []).append(int(i))
        for idxs in by_rat.values():
            if len(idxs) < 2:
                continue
            order = list(range(len(idxs)))
            rng.shuffle(order)
            data_norm[idxs] = data_norm[idxs][order]
            data_raw[idxs] = data_raw[idxs][order]
            if day_valid is not None:
                day_valid[idxs] = day_valid[idxs][order]
        gd = build_group_data(
            data_norm,
            n_chunks=8,
            feature_selection={"daily": ["mean", "range"]},
            data_raw=data_raw,
            n_points_per_day=24,
            fixed_n_days=8,
            day_valid=day_valid,
        )
        rec = _fit(
            gd,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            nuts_backend=args.nuts_backend,
        )
        rec["arm"] = "shuffle_dates_proxy_row_shuffle_within_rat"
        rec["seed"] = seed
        results.append(rec)
        print(
            f"[neg] seed={seed} E[τ]={rec['e_tau']:.3f} "
            f"elpd/fe={rec['elpd_loo_per_feature_event']:.3f}",
            flush=True,
        )

    out_json = args.out_dir / "neg_control_results.json"
    out_csv = args.out_dir / "neg_control_results.csv"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[neg] wrote {out_json}")
    print(f"[neg] wrote {out_csv}")


if __name__ == "__main__":
    main()
