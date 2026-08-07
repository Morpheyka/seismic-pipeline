#!/usr/bin/env python3
"""Density-safe confirmatory / sensitivity MCMC (publishable full-rerun rev. 3).

Primary A/B with day-mask ON, K=6; before_only; mask OFF sensitivity;
Pareto IDs without dropping windows. Uses re-exported nanpad profiles
(drop_incomplete_events=False) under --profile-cache or re-exports on demand.

Usage (from seismic_pipeline_standalone/):
  python scripts/run_density_safe_confirmatory.py --preset priority
  python scripts/run_density_safe_confirmatory.py --preset all --tune 2000 --draws 2000
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
from seismic_pipeline.bayesian.diagnostics import collect_pareto_k_stats, score_changepoint_trace
from seismic_pipeline.config import FULL_EXHAUSTIVE_EVENTS_8DAY, default_export_base_cfg
from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
from seismic_pipeline.features.runtime import set_runtime_export_cfg
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only
from seismic_pipeline.visualization.changepoint_ppc import (
    plot_posterior_predictive_check,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile-cache",
        type=Path,
        default=PROJECT_ROOT / "run_output_8day_density_safe" / "profile_cache",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "run_output_density_safe_confirmatory",
    )
    p.add_argument(
        "--preset",
        choices=("priority", "sensitivity", "all", "primary_a_only", "diag"),
        default="priority",
    )
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--nuts-backend", default="blackjax", choices=("blackjax", "pymc", "numpyro"))
    p.add_argument("--reexport", action="store_true", help="Force re-export profiles.")
    p.add_argument("--make-ppc", action="store_true", help="Write PPC figures for Primary A/B.")
    return p.parse_args()


def _profile_csv(cache_root: Path, n_points: int, overlap: float) -> Path:
    name = f"rem_n{int(n_points)}_ov{float(overlap):.2f}_stage2"
    return cache_root / name / "samples_10days_nanpad.csv"


def _ensure_profile(
    cache_root: Path,
    n_points: int,
    overlap: float,
    *,
    reexport: bool,
) -> Path:
    csv_path = _profile_csv(cache_root, n_points, overlap)
    if csv_path.is_file() and not reexport:
        return csv_path
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    export_cfg = default_export_base_cfg(output_dir=str(out_dir))
    export_cfg["events"] = [dict(x) for x in FULL_EXHAUSTIVE_EVENTS_8DAY]
    export_cfg["window_days"] = 8
    export_cfg["drop_incomplete_events"] = False
    export_cfg["n_points_per_day"] = int(n_points)
    export_cfg["overlap"] = float(overlap)
    export_cfg["rem_stage"] = 2
    export_rem_profiles_10days_cached_only(**export_cfg)
    return csv_path


def _exported_meta(csv_path: Path) -> pd.DataFrame:
    meta_path = csv_path.with_name(csv_path.name.replace("_nanpad.csv", "_metadata.csv"))
    if not meta_path.is_file():
        meta_path = csv_path.parent / "samples_10days_metadata.csv"
    meta = pd.read_csv(meta_path)
    return meta[meta["exported"].astype(bool)].sort_values("row_index").reset_index(drop=True)


def _range_entry(lik: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"likelihood": lik, "support_upper": 2.0}
    if lik == "interval_inflated_beta":
        entry["threshold"] = 0.9
    return entry


def _configs_for_preset(preset: str) -> list[dict[str, Any]]:
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
        apply_artifacts: bool = True,
        stratum: str = "full",
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
                "apply_artifacts": apply_artifacts,
                "stratum": stratum,
            }
        )

    # Primary A
    add(
        tag="primary_a_st_bc_mask_on",
        features=["mean", "range"],
        mean_lik="student_t",
        range_lik="beta_constrained",
        n=24,
        ov=0.5,
        role="Primary A",
    )
    if preset == "primary_a_only":
        return cells

    # Primary B: daily:range on (24,0)/(24,0.25)/(24,0.5) × BC + ZOIB
    for n, ov in [(24, 0.0), (24, 0.25), (24, 0.5)]:
        add(
            tag=f"primary_b_bc_n{n}_ov{ov}",
            features=["range"],
            mean_lik=None,
            range_lik="beta_constrained",
            n=n,
            ov=ov,
            role="Primary B",
        )
        add(
            tag=f"primary_b_zoib_n{n}_ov{ov}",
            features=["range"],
            mean_lik=None,
            range_lik="zero_inflated_beta",
            n=n,
            ov=ov,
            role="Primary B",
        )

    if preset in ("sensitivity", "all"):
        add(
            tag="primary_a_mask_off",
            features=["mean", "range"],
            mean_lik="student_t",
            range_lik="beta_constrained",
            n=24,
            ov=0.5,
            role="sensitivity_mask_off",
            apply_artifacts=False,
        )
        add(
            tag="primary_a_before_only",
            features=["mean", "range"],
            mean_lik="student_t",
            range_lik="beta_constrained",
            n=24,
            ov=0.5,
            role="before_only",
            stratum="before_only",
        )
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
            tag="mean_skew_n24_ov0.5",
            features=["mean"],
            mean_lik="skew_normal",
            range_lik=None,
            n=24,
            ov=0.5,
            role="sensitivity",
        )

    if preset in ("diag", "all"):
        add(
            tag="range_plain_beta_n24_ov0.5_DIAG",
            features=["range"],
            mean_lik=None,
            range_lik="beta",
            n=24,
            ov=0.5,
            role="diagnostic_only",
        )

    if preset == "priority":
        # Keep Primary A + Primary B at (24,0.5) BC+ZOIB only for speed.
        keep = {
            "primary_a_st_bc_mask_on",
            "primary_b_bc_n24_ov0.5",
            "primary_b_zoib_n24_ov0.5",
        }
        cells = [c for c in cells if c["tag"] in keep]

    seen: set[str] = set()
    uniq = []
    for c in cells:
        if c["tag"] in seen:
            continue
        seen.add(c["tag"])
        uniq.append(c)
    return uniq


def _bad_for_stratum(meta: pd.DataFrame, stratum: str) -> list[int]:
    if stratum == "full":
        return []
    bad: list[int] = []
    for i, row in meta.iterrows():
        direction = str(row.get("window_direction", ""))
        if stratum == "before_only" and direction != "before":
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


def _refit_one(
    *,
    csv_path: Path,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
    tune: int,
    draws: int,
    chains: int,
    seed: int,
    nuts_backend: str,
    out_dir: Path,
    make_ppc: bool,
) -> dict[str, Any]:
    n = int(cfg["n_points"])
    ov = float(cfg["overlap"])
    stratum = str(cfg.get("stratum", "full"))
    apply_artifacts = bool(cfg.get("apply_artifacts", True))
    extra_bad = _bad_for_stratum(meta, stratum)
    set_runtime_export_cfg({"n_points_per_day": n, "overlap": ov, "rem_stage": 2})
    prep = prepare_model_data(
        str(csv_path),
        bad_sample_indices=extra_bad,
        day_mask=True,
        apply_artifacts=apply_artifacts,
        min_valid_days=6,
        n_points_per_day=n,
        window_days=8,
    )
    feature_selection = cfg["feature_selection"]
    parameter_selection = cfg["parameter_selection"]
    group_data = build_group_data(
        prep["data_norm"],
        n_chunks=8,
        feature_selection=feature_selection,
        data_raw=prep["data_raw"],
        n_points_per_day=n,
        fixed_n_days=8,
        day_valid=prep.get("day_valid"),
    )
    model = build_changepoint_model(
        group_data,
        tau_lower=2,
        tau_upper=8,
        parameter_selection=parameter_selection,
        tau_mode="marginalized",
    )
    t0 = time.time()
    _ = seed
    sample_kwargs: dict[str, Any] = dict(
        draws=draws,
        tune=tune,
        chains=chains,
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
    trace = sample_model(model, **sample_kwargs)
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
    good = np.asarray(prep["good_indices"], dtype=int)
    mapped_idxs = []
    for li in pk_idxs:
        if 0 <= li < len(good):
            mapped_idxs.append(int(good[li]))
        else:
            mapped_idxs.append(int(li))
    worst_mapped = (
        int(good[pk_worst]) if pk_worst is not None and 0 <= pk_worst < len(good) else pk_worst
    )

    n_feat = sum(len(v) for v in feature_selection.values())
    n_events = int(next(iter(next(iter(group_data.values())).values())).shape[0])
    elpd = float(score.get("elpd_loo", float("nan")))
    e_tau = float(score.get("e_tau", float("nan")))
    hdi_w = float(score.get("tau_hdi_60_width", float("nan")))

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ppc_path = None
    if make_ppc and str(cfg["role"]).startswith("Primary"):
        try:
            import matplotlib.pyplot as plt

            ppc_path = fig_dir / f"{cfg['tag']}_ppc.png"
            plot_posterior_predictive_check(
                trace,
                model,
                group_data=group_data,
                parameter_selection=parameter_selection,
                random_seed=seed,
            )
            plt.savefig(ppc_path, dpi=150, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:  # noqa: BLE001
            ppc_path = f"ppc_failed: {exc}"
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                pass

    # Save tau posterior summary figure-ish numbers
    tau_path = fig_dir / f"{cfg['tag']}_tau_summary.json"
    tau_path.write_text(
        json.dumps(
            {
                "e_tau": e_tau,
                "hdi_60_width": hdi_w,
                "tau_mode": score.get("tau_mode"),
                "elpd_loo": elpd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "tag": cfg["tag"],
        "role": cfg["role"],
        "feature_block": cfg["feature_block"],
        "mean_lik": cfg.get("mean_lik"),
        "range_lik": cfg.get("range_lik"),
        "n_points": n,
        "overlap": ov,
        "stratum": stratum,
        "apply_artifacts": apply_artifacts,
        "day_mask": True,
        "min_valid_days": 6,
        "seed": seed,
        "tune": tune,
        "draws": draws,
        "chains": chains,
        "nuts_backend": nuts_backend,
        "n_events_fit": n_events,
        "n_masked_days": int(prep.get("n_masked_days", 0) or 0),
        "sample_seconds": round(sample_s, 2),
        "e_tau": e_tau,
        "tau_hdi_60_width": hdi_w,
        "elpd_loo": elpd,
        "elpd_loo_se": float(score.get("elpd_loo_se", float("nan"))),
        "elpd_loo_per_feature_event": elpd / max(1, n_feat * n_events),
        "r_hat_max": float(score.get("r_hat_max", float("nan"))),
        "ess_min_bulk": float(score.get("ess_min_bulk", float("nan"))),
        "n_divergences": int(score.get("n_divergences", 0)),
        "loo_pareto_k_max": float(pk_max) if pk_max is not None else float("nan"),
        "loo_pareto_k_n_high": int(pk_n),
        "loo_pareto_event_indices": mapped_idxs,
        "loo_pareto_events": _event_identities(meta, mapped_idxs),
        "loo_pareto_worst_event": _event_identities(
            meta, [worst_mapped] if worst_mapped is not None else []
        ),
        "windows_dropped": False,
        "ppc_path": str(ppc_path) if ppc_path else None,
        "tau_summary_path": str(tau_path),
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = _configs_for_preset(args.preset)
    results: list[dict[str, Any]] = []
    for cfg in cells:
        csv_path = _ensure_profile(
            args.profile_cache,
            cfg["n_points"],
            cfg["overlap"],
            reexport=bool(args.reexport),
        )
        meta = _exported_meta(csv_path)
        print(f"[conf] {cfg['tag']} n={cfg['n_points']} ov={cfg['overlap']} …", flush=True)
        rec = _refit_one(
            csv_path=csv_path,
            meta=meta,
            cfg=cfg,
            tune=args.tune,
            draws=args.draws,
            chains=args.chains,
            seed=args.seed,
            nuts_backend=args.nuts_backend,
            out_dir=args.out_dir,
            make_ppc=bool(args.make_ppc),
        )
        results.append(rec)
        print(
            f"[conf] {cfg['tag']} E[τ]={rec['e_tau']:.3f} "
            f"elpd/fe={rec['elpd_loo_per_feature_event']:.3f} "
            f"n={rec['n_events_fit']} masked_days={rec['n_masked_days']}",
            flush=True,
        )

    out_json = args.out_dir / "confirmatory_results.json"
    out_csv = args.out_dir / "confirmatory_results.csv"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.json_normalize(results).to_csv(out_csv, index=False)
    print(f"[conf] wrote {out_json}")
    print(f"[conf] wrote {out_csv}")


if __name__ == "__main__":
    main()
