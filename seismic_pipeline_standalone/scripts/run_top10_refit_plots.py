#!/usr/bin/env python3
"""Refit top-N exhaustive-search models with longer MCMC and save diagnostic plots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import (
    _available_varnames,
    feature_likelihood_profiles,
    summary_from_trace,
    tau_probabilities,
)
from seismic_pipeline.bayesian.parallel_search import ParallelSearchConfig, _configure_blas_threads
from seismic_pipeline.config import FULL_EXHAUSTIVE_EVENTS_8DAY, default_export_base_cfg
from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
from seismic_pipeline.mod.threading_config import configure_threading
from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only
from seismic_pipeline.visualization.changepoint_plots import (
    plot_posteriors_like_script,
    plot_trace_and_tau,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit top models and save plots.")
    parser.add_argument(
        "--csv",
        default="./run_output_8day_parallel_full/exhaustive_search_parallel.csv",
        help="Exhaustive search results CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="./run_output_8day_parallel_full/top10_refits",
        help="Directory for refit outputs and plots.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top eligible models.")
    parser.add_argument("--tune", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--nuts-backend", default="blackjax")
    parser.add_argument("--tau-lower", type=int, default=3)
    parser.add_argument("--tau-upper", type=int, default=8)
    parser.add_argument("--window-days", type=int, default=8)
    parser.add_argument(
        "--only-fingerprint",
        default=None,
        help="Optional single fingerprint to refit (for debugging).",
    )
    return parser.parse_args()


def _parse_removed_indices(raw: Any) -> list[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw == "[]":
            return []
        return [int(x) for x in json.loads(raw)]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return []


def _select_top_models(df: pd.DataFrame, *, top_n: int, fingerprint: str | None) -> pd.DataFrame:
    if fingerprint:
        sub = df[df["fingerprint"].astype(str) == str(fingerprint)]
        if sub.empty:
            raise ValueError(f"fingerprint not found in CSV: {fingerprint!r}")
        return sub.head(1)

    eligible = df[df["rank_eligible"] == True].copy()  # noqa: E712
    if eligible.empty:
        eligible = df[df["status"] == "ok"].copy()
    return (
        eligible.sort_values("elpd_loo_per_feature_day", ascending=False)
        .head(max(1, int(top_n)))
        .reset_index(drop=True)
    )


def _profile_key(n_points: int, overlap: float) -> tuple[int, float]:
    return int(n_points), round(float(overlap), 4)


def _export_profile_cache(
    *,
    export_base_cfg: dict[str, Any],
    cache_dir: Path,
    n_points: int,
    overlap: float,
    window_days: int,
) -> dict[str, Any]:
    export_cfg = dict(export_base_cfg)
    export_cfg.update(
        {
            "output_dir": str(cache_dir),
            "window_days": int(window_days),
            "n_points_per_day": int(n_points),
            "overlap": float(overlap),
        }
    )
    export_result = export_rem_profiles_10days_cached_only(**export_cfg)
    prep = prepare_model_data(csv_path=export_result["paths"]["nanpad_output_csv"])
    return {
        "data_norm": np.asarray(prep["data_norm"], dtype=float),
        "data_raw": np.asarray(prep["data_raw"], dtype=float),
        "good_indices": np.asarray(prep.get("good_indices", np.arange(prep["data_norm"].shape[0])), dtype=int),
        "csv_path": prep.get("csv_path", export_result["paths"]["nanpad_output_csv"]),
    }


def _build_trace_vars(trace, group_data: dict) -> list[str]:
    available = _available_varnames(trace)
    trace_vars: list[str] = []
    if "tau" in available:
        trace_vars.append("tau")
    if "tau_mean" in available:
        trace_vars.append("tau_mean")
    for group_name, features in group_data.items():
        for feat_name in features.keys():
            for param_name in ("mu", "sigma", "alpha", "beta", "pi", "nu"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in available and p2 in available:
                    trace_vars.extend([p1, p2])
            # Shared Student-t degrees of freedom (legacy / default).
            nu_shared = f"nu_{group_name}_{feat_name}"
            if nu_shared in available:
                trace_vars.append(nu_shared)
    return trace_vars


def _refit_and_plot_one(
    *,
    row: pd.Series,
    data_bundle: dict[str, Any],
    out_dir: Path,
    tune: int,
    draws: int,
    chains: int,
    nuts_backend: str,
    tau_lower: int,
    tau_upper: int,
    window_days: int,
) -> dict[str, Any]:
    cfg = json.loads(row["config_json"])
    fingerprint = str(row["fingerprint"])
    rank = int(row.get("rank_by_loo", -1))
    stem = f"rank{rank:02d}_{fingerprint}"
    model_dir = out_dir / stem
    model_dir.mkdir(parents=True, exist_ok=True)

    removed = _parse_removed_indices(row.get("removed_event_indices"))
    active_idx = np.arange(int(data_bundle["data_norm"].shape[0]), dtype=int)
    if removed:
        mask = np.ones(active_idx.size, dtype=bool)
        for idx in removed:
            if 0 <= int(idx) < mask.size:
                mask[int(idx)] = False
        active_idx = active_idx[mask]

    n_points = int((cfg.get("rem_profile_params") or {}).get("n_points_per_day", row.get("n_points", 24)))
    data_norm = data_bundle["data_norm"][active_idx]
    data_raw = data_bundle["data_raw"][active_idx]

    group_data = build_group_data(
        data_norm,
        n_chunks=int(cfg["n_chunks"]),
        feature_selection=cfg["feature_selection"],
        data_raw=data_raw,
        window_days=int(window_days),
        n_points_per_day=n_points,
    )
    model = build_changepoint_model(
        group_data,
        tau_lower=int(tau_lower),
        tau_upper=int(tau_upper),
        parameter_selection=cfg["parameter_selection"],
        tau_mode="marginalized",
    )

    sample_kwargs: dict[str, Any] = {
        "draws": int(draws),
        "tune": int(tune),
        "nuts_backend": str(nuts_backend).strip().lower(),
        "chains": int(chains),
        "progressbar": False,
    }
    if sample_kwargs["nuts_backend"] == "blackjax":
        sample_kwargs["jax_chain_method"] = "parallel"

    print(f"[refit] {stem} features={row.get('features')} tune={tune} draws={draws}", flush=True)
    trace = sample_model(model, **sample_kwargs)

    trace_vars = _build_trace_vars(trace, group_data)
    summary_vars = [v for v in trace_vars if v in _available_varnames(trace)]
    try:
        summary = summary_from_trace(trace, summary_vars) if summary_vars else pd.DataFrame()
    except Exception as exc:
        print(f"[refit] warning: posterior summary failed: {exc}", flush=True)
        summary = pd.DataFrame()
    support, probs = tau_probabilities(trace)
    map_idx = int(np.argmax(probs))

    meta = {
        "fingerprint": fingerprint,
        "rank_by_loo": rank,
        "features": str(row.get("features", "")),
        "likelihoods": str(row.get("likelihoods", "")),
        "n_points": n_points,
        "overlap": float((cfg.get("rem_profile_params") or {}).get("overlap", row.get("overlap", 0.0))),
        "n_model_events": int(data_norm.shape[0]),
        "removed_event_indices": removed,
        "tune": int(tune),
        "draws": int(draws),
        "chains": int(chains),
        "tau_map": int(support[map_idx]),
        "tau_map_concentration": float(probs[map_idx]),
        "config": cfg,
    }
    (model_dir / "refit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not summary.empty:
        summary.to_csv(model_dir / "posterior_summary.csv")

    # Persist ArviZ DataTree for offline diagnostic replotting.
    try:
        trace_nc = model_dir / "trace.nc"
        trace.to_netcdf(str(trace_nc))
        print(f"[refit] wrote {trace_nc}", flush=True)
    except Exception as exc:
        print(f"[refit] warning: failed to save trace.nc: {exc}", flush=True)
        try:
            trace_zarr = model_dir / "trace.zarr"
            if trace_zarr.exists():
                import shutil

                shutil.rmtree(trace_zarr)
            trace.to_zarr(str(trace_zarr))
            print(f"[refit] wrote {trace_zarr}", flush=True)
        except Exception as exc2:
            print(f"[refit] warning: failed to save trace.zarr: {exc2}", flush=True)

    # Persist observed feature arrays for likelihood overlays.
    try:
        obs_payload: dict[str, Any] = {}
        for group_name, features in group_data.items():
            for feat_name, observed_df in features.items():
                key = f"{group_name}__{feat_name}"
                obs_payload[key] = np.asarray(observed_df.to_numpy(dtype=float))
        np.savez_compressed(model_dir / "observations.npz", **obs_payload)
        print(f"[refit] wrote {model_dir / 'observations.npz'}", flush=True)
    except Exception as exc:
        print(f"[refit] warning: failed to save observations.npz: {exc}", flush=True)

    title = f"rank {rank} {fingerprint} | {row.get('features', '')}"
    plot_trace_and_tau(
        trace,
        trace_vars,
        title_prefix=title,
        save_path=model_dir / "trace.png",
    )
    plot_posteriors_like_script(
        trace,
        group_data=group_data,
        title_prefix=title,
        save_path=model_dir / "posteriors.png",
        tau_bar_save_path=model_dir / "tau_posterior.png",
    )
    try:
        feature_likelihood_profiles(
            trace,
            group_data=group_data,
            parameter_selection=cfg["parameter_selection"],
            grid_size=300,
            plot=True,
            save_dir=model_dir,
            plot_stem_prefix=stem,
        )
    except Exception as exc:
        print(f"[refit] warning: likelihood profile plots failed: {exc}", flush=True)
    print(f"[refit] saved plots to {model_dir}", flush=True)
    return meta


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_threading(cores=16, threads_per_job=max(1, 10 // 4))
    blas_cfg = ParallelSearchConfig(
        n_jobs=1,
        chains=int(args.chains),
        cores_per_chain=2,
        blas_threads_per_worker=2,
        nuts_backend=str(args.nuts_backend),
    )
    _configure_blas_threads(blas_cfg)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    top = _select_top_models(df, top_n=args.top_n, fingerprint=args.only_fingerprint)
    print(f"[select] {len(top)} model(s) to refit", flush=True)

    export_base_cfg = default_export_base_cfg(output_dir=str(out_dir / "profile_cache"))
    export_base_cfg["events"] = [dict(x) for x in FULL_EXHAUSTIVE_EVENTS_8DAY]
    export_base_cfg["window_days"] = int(args.window_days)
    export_base_cfg["drop_incomplete_events"] = True

    profile_cache: dict[tuple[int, float], dict[str, Any]] = {}
    metas: list[dict[str, Any]] = []

    for _, row in top.iterrows():
        cfg = json.loads(row["config_json"])
        rem = cfg.get("rem_profile_params") or {}
        n_points = int(rem.get("n_points_per_day", row.get("n_points", 24)))
        overlap = float(rem.get("overlap", row.get("overlap", 0.0)))
        key = _profile_key(n_points, overlap)
        if key not in profile_cache:
            cache_dir = out_dir / "profile_cache" / f"n{n_points}_ov{overlap:.2f}"
            print(f"[export] profile n_points={n_points} overlap={overlap}", flush=True)
            profile_cache[key] = _export_profile_cache(
                export_base_cfg=export_base_cfg,
                cache_dir=cache_dir,
                n_points=n_points,
                overlap=overlap,
                window_days=int(args.window_days),
            )

        meta = _refit_and_plot_one(
            row=row,
            data_bundle=profile_cache[key],
            out_dir=out_dir,
            tune=int(args.tune),
            draws=int(args.draws),
            chains=int(args.chains),
            nuts_backend=str(args.nuts_backend),
            tau_lower=int(args.tau_lower),
            tau_upper=int(args.tau_upper),
            window_days=int(args.window_days),
        )
        metas.append(meta)

    summary_path = out_dir / "refit_summary.json"
    summary_path.write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
