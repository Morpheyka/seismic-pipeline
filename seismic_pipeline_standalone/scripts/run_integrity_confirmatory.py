#!/usr/bin/env python3
"""Integrity confirmatory / sensitivity MCMC for research_integrity_checklist.

Runs Primary A/B and τ-sensitivity cells from cached July profile CSVs
(same builder as parallel_search). Prefer beta_constrained / IIB — never
claim plain beta as a winner.

Usage (from seismic_pipeline_standalone/):
  python scripts/run_integrity_confirmatory.py
  python scripts/run_integrity_confirmatory.py --preset priority
  python scripts/run_integrity_confirmatory.py --preset sensitivity --tune 300 --draws 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys_path_insert = PROJECT_ROOT
import sys

sys.path.insert(0, str(sys_path_insert))

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import collect_pareto_k_stats, score_changepoint_trace
from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
from seismic_pipeline.features.runtime import set_runtime_export_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile-cache",
        type=Path,
        default=PROJECT_ROOT / "run_output_8day_parallel_full" / "profile_cache",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "run_output_integrity_confirmatory",
    )
    p.add_argument(
        "--preset",
        choices=("priority", "sensitivity", "all", "primary_a_only"),
        default="priority",
        help="Which config grid to run.",
    )
    p.add_argument("--tune", type=int, default=500)
    p.add_argument("--draws", type=int, default=400)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--nuts-backend", default="blackjax", choices=("blackjax", "pymc", "numpyro"))
    p.add_argument(
        "--stratum",
        default="full",
        choices=("full", "before_only", "drop_2024_09_30"),
        help="Event stratum applied via bad_sample_indices on exported rows.",
    )
    return p.parse_args()


def _profile_csv(cache_root: Path, n_points: int, overlap: float) -> Path:
    name = f"rem_n{int(n_points)}_ov{float(overlap):.2f}_stage2"
    return cache_root / name / "samples_10days_nanpad.csv"


def _exported_meta(csv_path: Path) -> pd.DataFrame:
    meta_path = csv_path.with_name(csv_path.name.replace("_nanpad.csv", "_metadata.csv"))
    if not meta_path.is_file():
        meta_path = csv_path.parent / "samples_10days_metadata.csv"
    meta = pd.read_csv(meta_path)
    return meta[meta["exported"].astype(bool)].sort_values("row_index").reset_index(drop=True)


def _bad_indices_for_stratum(meta: pd.DataFrame, stratum: str) -> list[int]:
    if stratum == "full":
        return []
    bad: list[int] = []
    for i, row in meta.iterrows():
        date = str(row.get("event_date", ""))
        direction = str(row.get("window_direction", ""))
        if stratum == "before_only" and direction != "before":
            bad.append(int(i))
        elif stratum == "drop_2024_09_30" and date == "2024-09-30":
            bad.append(int(i))
    return bad


def _event_identities(meta: pd.DataFrame, indices: list[int]) -> list[dict[str, Any]]:
    out = []
    for idx in indices:
        if idx < 0 or idx >= len(meta):
            out.append({"matrix_idx": idx, "error": "out_of_range"})
            continue
        row = meta.iloc[idx]
        out.append(
            {
                "matrix_idx": int(idx),
                "rat_id": str(row["rat_id"]),
                "event_date": str(row["event_date"]),
                "direction": str(row["window_direction"]),
            }
        )
    return out


def _range_entry(lik: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"likelihood": lik, "support_upper": 2.0}
    if lik == "interval_inflated_beta":
        entry["threshold"] = 0.9
    return entry


def _configs_for_preset(preset: str) -> list[dict[str, Any]]:
    """Protocol cells from checklist §2; priority = Primary A + B subset + key bridges."""
    cells: list[dict[str, Any]] = []

    def add(
        *,
        tag: str,
        features: list[str],
        mean_lik: str | None,
        range_lik: str | None,
        n: int,
        ov: float,
        role: str,
    ) -> None:
        fs = {"daily": list(features)}
        ps: dict[str, Any] = {}
        if "mean" in features and mean_lik:
            ps["mean"] = {"likelihood": mean_lik}
        if "range" in features and range_lik:
            ps["range"] = _range_entry(range_lik)
        cells.append(
            {
                "tag": tag,
                "role": role,
                "feature_selection": fs,
                "parameter_selection": ps,
                "n_points": n,
                "overlap": ov,
                "mean_lik": mean_lik,
                "range_lik": range_lik,
                "feature_block": "daily:" + "+".join(features),
            }
        )

    # Primary A
    add(
        tag="primary_a_st_bc",
        features=["mean", "range"],
        mean_lik="student_t",
        range_lik="beta_constrained",
        n=24,
        ov=0.5,
        role="Primary A",
    )
    if preset == "primary_a_only":
        return cells

    # Primary B confirmatory trio (subset for priority; all three for sensitivity/all)
    b_grids = [(48, 0.0), (48, 0.25), (24, 0.5)]
    if preset == "priority":
        b_grids = [(24, 0.5), (48, 0.0)]  # one Primary-A grid twin + classic July grid
    for n, ov in b_grids:
        add(
            tag=f"primary_b_bc_n{n}_ov{ov}",
            features=["range"],
            mean_lik=None,
            range_lik="beta_constrained",
            n=n,
            ov=ov,
            role="Primary B",
        )

    if preset in ("sensitivity", "all"):
        # daily:mean student_t grid
        for n, ov in [(12, 0.0), (24, 0.0), (24, 0.5), (48, 0.0)]:
            add(
                tag=f"mean_st_n{n}_ov{ov}",
                features=["mean"],
                mean_lik="student_t",
                range_lik=None,
                n=n,
                ov=ov,
                role="sensitivity",
            )
        add(
            tag="mean_normal_n24_ov0.5",
            features=["mean"],
            mean_lik="normal",
            range_lik=None,
            n=24,
            ov=0.5,
            role="sensitivity",
        )
        # daily:range beta_constrained remaining grids
        for n, ov in [(12, 0.0), (24, 0.0), (48, 0.25)]:
            if (n, ov) in b_grids:
                continue
            add(
                tag=f"range_bc_n{n}_ov{ov}",
                features=["range"],
                mean_lik=None,
                range_lik="beta_constrained",
                n=n,
                ov=ov,
                role="sensitivity",
            )
        # IIB cells
        for n, ov in [(24, 0.5), (48, 0.0)]:
            add(
                tag=f"range_iib_n{n}_ov{ov}",
                features=["range"],
                mean_lik=None,
                range_lik="interval_inflated_beta",
                n=n,
                ov=ov,
                role="sensitivity",
            )
        # diagnostic plain beta (not a winner)
        add(
            tag="range_plain_beta_n48_ov0_DIAG",
            features=["range"],
            mean_lik=None,
            range_lik="beta",
            n=48,
            ov=0.0,
            role="diagnostic_only",
        )
        # mean+range variants
        add(
            tag="meanrange_st_iib_n24_ov0.5",
            features=["mean", "range"],
            mean_lik="student_t",
            range_lik="interval_inflated_beta",
            n=24,
            ov=0.5,
            role="sensitivity",
        )
        add(
            tag="meanrange_normal_bc_n24_ov0.5",
            features=["mean", "range"],
            mean_lik="normal",
            range_lik="beta_constrained",
            n=24,
            ov=0.5,
            role="sensitivity",
        )
        add(
            tag="meanrange_st_bc_n48_ov0",
            features=["mean", "range"],
            mean_lik="student_t",
            range_lik="beta_constrained",
            n=48,
            ov=0.0,
            role="sensitivity",
        )
    elif preset == "priority":
        # bridge cells for Primary A alternatives
        add(
            tag="meanrange_normal_bc_n24_ov0.5",
            features=["mean", "range"],
            mean_lik="normal",
            range_lik="beta_constrained",
            n=24,
            ov=0.5,
            role="Primary A alt",
        )
        add(
            tag="mean_st_n24_ov0.5",
            features=["mean"],
            mean_lik="student_t",
            range_lik=None,
            n=24,
            ov=0.5,
            role="sensitivity",
        )

    if preset == "all":
        # ensure Primary B full trio already added in sensitivity path
        pass

    # de-dupe by tag
    seen: set[str] = set()
    uniq = []
    for c in cells:
        if c["tag"] in seen:
            continue
        seen.add(c["tag"])
        uniq.append(c)
    return uniq


def _refit_one(
    *,
    csv_path: Path,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
    stratum: str,
    tune: int,
    draws: int,
    chains: int,
    seed: int,
    nuts_backend: str,
) -> dict[str, Any]:
    n = int(cfg["n_points"])
    ov = float(cfg["overlap"])
    bad = _bad_indices_for_stratum(meta, stratum)
    set_runtime_export_cfg({"n_points_per_day": n, "overlap": ov, "rem_stage": 2})
    prep = prepare_model_data(str(csv_path), bad_sample_indices=bad)
    feature_selection = cfg["feature_selection"]
    parameter_selection = cfg["parameter_selection"]
    group_data = build_group_data(
        prep["data_norm"],
        n_chunks=8,
        feature_selection=feature_selection,
        data_raw=prep["data_raw"],
        n_points_per_day=n,
        fixed_n_days=8,
    )
    model = build_changepoint_model(
        group_data,
        tau_lower=3,
        tau_upper=8,
        parameter_selection=parameter_selection,
        tau_mode="marginalized",
    )
    t0 = time.time()
    # sample_model has no random_seed kw; seed is logged for protocol reproducibility notes.
    _ = seed
    trace = sample_model(
        model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
        progressbar=False,
        nuts_backend=nuts_backend,
    )
    sample_s = time.time() - t0
    score = score_changepoint_trace(
        trace,
        group_data=group_data,
        parameter_selection=parameter_selection,
        model=model,
        criterion="loo",
        warn_on_fallback=False,
        loo_report="elpd",
    )
    loo_obj = score.get("_loo_obj")
    pk_max, pk_n, pk_idxs, pk_worst = collect_pareto_k_stats(
        trace, model, loo_obj=loo_obj
    )
    # Map Pareto indices into the *fitted* event set (after bad drops).
    # prepare_model_data remaps rows; LOO indices are over remaining events 0..n_fit-1.
    # Identity mapping: good_indices[loo_idx] -> original exported matrix index.
    good = np.asarray(prep["good_indices"], dtype=int)
    mapped_idxs = []
    for li in pk_idxs:
        if 0 <= li < len(good):
            mapped_idxs.append(int(good[li]))
        else:
            mapped_idxs.append(int(li))
    worst_mapped = int(good[pk_worst]) if pk_worst is not None and 0 <= pk_worst < len(good) else pk_worst

    n_feat = sum(len(v) for v in feature_selection.values())
    n_events = int(next(iter(next(iter(group_data.values())).values())).shape[0])
    elpd = float(score.get("elpd_loo", float("nan")))
    e_tau = float(score.get("e_tau", float("nan")))
    hdi_w = float(score.get("tau_hdi_60_width", float("nan")))
    # Unidentifiable flag: mass near prior edges with flat posterior
    unident = bool(
        (e_tau <= 3.3 or e_tau >= 7.7) and (not np.isfinite(hdi_w) or hdi_w >= 3.5)
    )

    return {
        "tag": cfg["tag"],
        "role": cfg["role"],
        "feature_block": cfg["feature_block"],
        "mean_lik": cfg.get("mean_lik"),
        "range_lik": cfg.get("range_lik"),
        "likelihoods": {k: v.get("likelihood") for k, v in parameter_selection.items()},
        "n_points": n,
        "overlap": ov,
        "stratum": stratum,
        "seed": seed,
        "tune": tune,
        "draws": draws,
        "chains": chains,
        "nuts_backend": nuts_backend,
        "n_events_fit": n_events,
        "n_events_exported": int(len(meta)),
        "n_features": n_feat,
        "dropped_matrix_indices": bad,
        "dropped_identities": _event_identities(meta, bad),
        "elpd_loo": elpd,
        "loo_ic": float(score.get("loo_ic", float("nan"))),
        "elpd_loo_per_feature_event": (
            elpd / float(n_feat * n_events) if n_feat and n_events else float("nan")
        ),
        "e_tau": e_tau,
        "tau_hdi_60_lower": float(score.get("tau_hdi_60_lower", float("nan"))),
        "tau_hdi_60_upper": float(score.get("tau_hdi_60_upper", float("nan"))),
        "tau_hdi_60_width": hdi_w,
        "tau_q2": float(score.get("tau_q2", float("nan"))),
        "r_hat_max": float(score.get("r_hat_max", float("nan"))),
        "ess_min_bulk": float(score.get("ess_min_bulk", float("nan"))),
        "n_divergences": int(score.get("n_divergences", 0)),
        "pareto_k_max": float(pk_max),
        "pareto_k_n_over_0.7": int(pk_n),
        "pareto_over_loo_indices": list(pk_idxs),
        "pareto_over_matrix_indices": mapped_idxs,
        "pareto_over_identities": _event_identities(meta, mapped_idxs),
        "pareto_worst_matrix_idx": worst_mapped,
        "unidentifiable_tau": unident,
        "sample_seconds": float(sample_s),
        "includes_2024_09_30": _includes_2024_09_30(meta, stratum, bad),
    }


def _includes_2024_09_30(meta: pd.DataFrame, stratum: str, bad: list[int]) -> bool:
    bad_set = set(bad)
    for i, row in meta.iterrows():
        if int(i) in bad_set:
            continue
        if str(row["event_date"]) != "2024-09-30":
            continue
        if stratum == "before_only" and str(row["window_direction"]) != "before":
            continue
        return True
    return False


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = _configs_for_preset(args.preset)
    results: list[dict[str, Any]] = []
    print(
        f"[integrity] preset={args.preset} stratum={args.stratum} "
        f"n_configs={len(configs)} tune={args.tune} draws={args.draws} chains={args.chains}",
        flush=True,
    )

    for i, cfg in enumerate(configs):
        csv_path = _profile_csv(args.profile_cache, cfg["n_points"], cfg["overlap"])
        print(
            f"[{i+1}/{len(configs)}] {cfg['tag']} N={cfg['n_points']} ov={cfg['overlap']} "
            f"{cfg['feature_block']} …",
            flush=True,
        )
        if not csv_path.is_file():
            results.append({"tag": cfg["tag"], "error": f"missing {csv_path}", **cfg})
            print(f"  ERROR missing cache {csv_path}", flush=True)
            continue
        meta = _exported_meta(csv_path)
        try:
            row = _refit_one(
                csv_path=csv_path,
                meta=meta,
                cfg=cfg,
                stratum=args.stratum,
                tune=args.tune,
                draws=args.draws,
                chains=args.chains,
                seed=args.seed,
                nuts_backend=args.nuts_backend,
            )
            results.append(row)
            print(
                f"  E[τ]={row['e_tau']:.3f} HDI60w={row['tau_hdi_60_width']:.3f} "
                f"elpd/fe={row['elpd_loo_per_feature_event']:.3f} "
                f"pareto_k_max={row['pareto_k_max']:.3f} n_over={row['pareto_k_n_over_0.7']} "
                f"({row['sample_seconds']:.1f}s)",
                flush=True,
            )
            if row["pareto_over_identities"]:
                ids = ", ".join(
                    f"{x.get('rat_id')}/{x.get('event_date')}/{x.get('direction')}"
                    for x in row["pareto_over_identities"]
                )
                print(f"  Pareto>0.7: {ids}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"tag": cfg["tag"], "error": str(exc), **cfg})
            print(f"  ERROR: {exc}", flush=True)

    out_json = args.out_dir / f"integrity_{args.preset}_{args.stratum}_seed{args.seed}.json"
    payload = {
        "preset": args.preset,
        "stratum": args.stratum,
        "seed": args.seed,
        "tune": args.tune,
        "draws": args.draws,
        "chains": args.chains,
        "nuts_backend": args.nuts_backend,
        "builder": "July profile_cache rem_n*_ov*_stage2 + prepare_model_data/build_group_data",
        "note": "Plain beta only as diagnostic_only; primary uses beta_constrained/IIB.",
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Flat CSV for sensitivity table
    flat_rows = []
    for r in results:
        if "error" in r and "e_tau" not in r:
            flat_rows.append(
                {
                    "tag": r.get("tag"),
                    "role": r.get("role"),
                    "feature_block": r.get("feature_block"),
                    "error": r.get("error"),
                }
            )
            continue
        flat_rows.append(
            {
                "tag": r.get("tag"),
                "role": r.get("role"),
                "feature_block": r.get("feature_block"),
                "mean_lik": r.get("mean_lik"),
                "range_lik": r.get("range_lik"),
                "n_points": r.get("n_points"),
                "overlap": r.get("overlap"),
                "stratum": r.get("stratum"),
                "e_tau": r.get("e_tau"),
                "tau_hdi_60_width": r.get("tau_hdi_60_width"),
                "elpd_loo": r.get("elpd_loo"),
                "elpd_loo_per_feature_event": r.get("elpd_loo_per_feature_event"),
                "loo_ic": r.get("loo_ic"),
                "pareto_k_max": r.get("pareto_k_max"),
                "pareto_k_n_over_0.7": r.get("pareto_k_n_over_0.7"),
                "n_events_fit": r.get("n_events_fit"),
                "includes_2024_09_30": r.get("includes_2024_09_30"),
                "unidentifiable_tau": r.get("unidentifiable_tau"),
                "r_hat_max": r.get("r_hat_max"),
                "seed": r.get("seed"),
                "tune": r.get("tune"),
                "draws": r.get("draws"),
                "chains": r.get("chains"),
            }
        )
    out_csv = args.out_dir / f"integrity_{args.preset}_{args.stratum}_seed{args.seed}.csv"
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
