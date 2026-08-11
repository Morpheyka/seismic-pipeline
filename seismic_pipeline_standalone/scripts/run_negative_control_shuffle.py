#!/usr/bin/env python3
"""Negative control: shuffle event dates / permute labels for changepoint integrity.

Under the null (no shared pre-event structure), τ should approach the DiscreteUniform
prior on {tau_lower…tau_upper} and ELPD should worsen vs the real-date primary config.

Default is a cheap dry-run: build shuffled event lists, write JSON, print success criteria.
Optional --run-mcmc fits one primary-like config on shuffled events (expensive; needs
local profile/export data like other search scripts).

Usage (from seismic_pipeline_standalone/):
  python scripts/run_negative_control_shuffle.py --dry-run
  python scripts/run_negative_control_shuffle.py --mode shuffle_dates --seed 0 --dry-run
  python scripts/run_negative_control_shuffle.py --mode permute_labels --seed 1 --out-dir ./neg_ctrl
  python scripts/run_negative_control_shuffle.py --run-mcmc --feature daily:mean+range \\
      --n-points 24 --overlap 0.5 --range-likelihood beta_constrained

See reports/research_integrity_checklist.md §5.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_default_events() -> list[dict[str, Any]]:
    """Load FULL_EXHAUSTIVE_EVENTS_8DAY without requiring sklearn/numpy package import."""
    try:
        from seismic_pipeline.config.changepoint_defaults import FULL_EXHAUSTIVE_EVENTS_8DAY

        return [dict(x) for x in FULL_EXHAUSTIVE_EVENTS_8DAY]
    except Exception:  # noqa: BLE001 — dry-run must work in bare python3
        import ast
        import re

        path = (
            PROJECT_ROOT
            / "seismic_pipeline"
            / "config"
            / "changepoint_defaults.py"
        )
        src = path.read_text(encoding="utf-8")
        match = re.search(
            r"FULL_EXHAUSTIVE_EVENTS_8DAY(?:\s*:\s*[^=]+)?\s*=\s*(\[[\s\S]*?\n\])",
            src,
        )
        if not match:
            raise RuntimeError(f"Could not parse event list from {path}") from None
        return [dict(x) for x in ast.literal_eval(match.group(1))]


SUCCESS_CRITERIA = """
Success criteria (vs real-date primary fit, same MCMC budget / builder):
  - E[τ] moves toward prior center (~5.5 if DiscreteUniform on 3..8) or HDI widens sharply
  - elpd_loo_per_feature_event clearly lower than real-date primary
  - Do not "rescue" Pareto by dropping events unless the real arm uses the same drops

Fail (worry about spurious real-date τ):
  - E[τ] stays sharply ~6.5–7.5 with narrow HDI under shuffle
  - Shuffled ELPD ≥ real ELPD
""".strip()


def shuffle_dates_within_rat(
    events: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Permute ``date`` among events that share the same ``rat_id`` (keeps direction)."""
    rng = random.Random(seed)
    out = [dict(e) for e in events]
    by_rat: dict[str, list[int]] = {}
    for i, e in enumerate(out):
        by_rat.setdefault(str(e["rat_id"]), []).append(i)
    for idxs in by_rat.values():
        dates = [out[i]["date"] for i in idxs]
        perm = list(range(len(dates)))
        rng.shuffle(perm)
        for j, i in enumerate(idxs):
            out[i]["date"] = dates[perm[j]]
    return out


def permute_event_labels(
    events: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Globally permute (date, direction) pairs across the event list."""
    rng = random.Random(seed)
    out = [dict(e) for e in events]
    labels = [(e.get("date"), e.get("direction")) for e in out]
    perm = list(range(len(labels)))
    rng.shuffle(perm)
    for i, e in enumerate(out):
        date, direction = labels[perm[i]]
        e["date"] = date
        if direction is not None:
            e["direction"] = direction
        elif "direction" in e:
            del e["direction"]
    return out


def summarize_shuffle(
    original: list[dict[str, Any]],
    shuffled: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(original)
    date_changed = sum(1 for a, b in zip(original, shuffled) if a.get("date") != b.get("date"))
    dir_changed = sum(
        1 for a, b in zip(original, shuffled) if a.get("direction") != b.get("direction")
    )
    return {
        "n_events": n,
        "n_date_changed": date_changed,
        "n_direction_changed": dir_changed,
        "frac_date_changed": float(date_changed / n) if n else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("shuffle_dates", "permute_labels"),
        default="shuffle_dates",
        help="Nullification mode (default: shuffle_dates within rat).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "run_output_negative_control",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Write shuffled events + protocol only (default).",
    )
    p.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Allow --run-mcmc path (still requires --run-mcmc to fit).",
    )
    p.add_argument(
        "--run-mcmc",
        action="store_true",
        help="Fit one config on shuffled events (expensive; needs local data).",
    )
    p.add_argument(
        "--feature",
        default="daily:mean+range",
        choices=("daily:mean", "daily:range", "daily:mean+range"),
        help="Primary-like feature block for optional MCMC.",
    )
    p.add_argument("--n-points", type=int, default=24)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument(
        "--range-likelihood",
        default="beta_constrained",
        choices=("beta_constrained", "interval_inflated_beta", "beta"),
        help="Range likelihood; prefer beta_constrained/IIB for integrity primary.",
    )
    p.add_argument("--mean-likelihood", default="student_t", choices=("student_t", "normal"))
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--tau-lower", type=int, default=3)
    p.add_argument("--tau-upper", type=int, default=8)
    p.add_argument(
        "--before-only",
        action="store_true",
        help="Restrict to direction=before events before shuffling.",
    )
    return p.parse_args()


def _feature_selection(feature: str) -> dict[str, list[str]]:
    if feature == "daily:mean":
        return {"daily": ["mean"]}
    if feature == "daily:range":
        return {"daily": ["range"]}
    return {"daily": ["mean", "range"]}


def _parameter_selection(
    feature: str,
    *,
    mean_lik: str,
    range_lik: str,
) -> dict[str, dict[str, Any]]:
    sel: dict[str, dict[str, Any]] = {}
    feats = _feature_selection(feature)
    metrics = feats.get("daily", [])
    if "mean" in metrics:
        sel["mean"] = {"likelihood": mean_lik}
    if "range" in metrics:
        cfg: dict[str, Any] = {"likelihood": range_lik, "support_upper": 2.0}
        if range_lik == "interval_inflated_beta":
            cfg["threshold"] = 0.9
        sel["range"] = cfg
    return sel


def _fit_group_data(
    group_data: Any,
    *,
    args: argparse.Namespace,
    feature_selection: dict[str, list[str]],
    parameter_selection: dict[str, dict[str, Any]],
    nuts_backend: str = "blackjax",
) -> dict[str, Any]:
    from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
    from seismic_pipeline.bayesian.diagnostics import score_changepoint_trace

    print(
        f"[neg-ctrl] Sampling MCMC tune={args.tune} draws={args.draws} "
        f"chains={args.chains} backend={nuts_backend}…"
    )
    model = build_changepoint_model(
        group_data,
        parameter_selection=parameter_selection,
        tau_lower=int(args.tau_lower),
        tau_upper=int(args.tau_upper),
        tau_mode="marginalized",
    )
    idata = sample_model(
        model,
        tune=int(args.tune),
        draws=int(args.draws),
        chains=int(args.chains),
        cores=1,
        progressbar=False,
        nuts_backend=nuts_backend,
    )
    scores = score_changepoint_trace(
        idata,
        group_data=group_data,
        parameter_selection=parameter_selection,
        model=model,
        criterion="loo",
        warn_on_fallback=False,
        loo_report="elpd",
    )
    n_feat = sum(len(v) for v in feature_selection.values())
    n_events = int(next(iter(next(iter(group_data.values())).values())).shape[0])
    elpd = float(scores.get("elpd_loo", float("nan")))
    return {
        "e_tau": float(scores.get("e_tau", float("nan"))),
        "tau_mean": float(scores.get("e_tau", float("nan"))),  # alias for older readers
        "tau_hdi_60_width": float(scores.get("tau_hdi_60_width", float("nan"))),
        "elpd_loo": elpd,
        "elpd_loo_per_feature_event": (
            elpd / float(n_feat * n_events) if n_feat and n_events else float("nan")
        ),
        "loo_ic": float(scores.get("loo_ic", float("nan"))),
        "n_events": n_events,
        "n_features": n_feat,
        "r_hat_max": float(scores.get("r_hat_max", float("nan"))),
    }


def _row_shuffle_within_rat_null(
    args: argparse.Namespace,
    out_dir: Path,
    feature_selection: dict[str, list[str]],
    parameter_selection: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Proxy null: permute profile rows within rat on cached Primary-A CSV.

    Used when calendar rebuild for shuffled dates cannot export (missing hypnograms).
    Preserves within-rat profile marginals but breaks event↔window alignment.
    """
    import numpy as np
    import pandas as pd

    from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
    from seismic_pipeline.features.runtime import set_runtime_export_cfg

    cache = (
        PROJECT_ROOT
        / "run_output_8day_parallel_full"
        / "profile_cache"
        / f"rem_n{int(args.n_points)}_ov{float(args.overlap):.2f}_stage2"
        / "samples_10days_nanpad.csv"
    )
    if not cache.is_file():
        return {"status": "skipped", "reason": f"proxy cache missing: {cache}"}

    meta_path = cache.parent / "samples_10days_metadata.csv"
    meta = pd.read_csv(meta_path)
    exported = meta[meta["exported"].astype(bool)].sort_values("row_index").reset_index(drop=True)
    set_runtime_export_cfg(
        {"n_points_per_day": int(args.n_points), "overlap": float(args.overlap), "rem_stage": 2}
    )
    prep = prepare_model_data(str(cache))
    data_norm = np.asarray(prep["data_norm"], dtype=float).copy()
    data_raw = np.asarray(prep["data_raw"], dtype=float).copy()
    if len(exported) != data_norm.shape[0]:
        return {
            "status": "failed",
            "reason": f"meta exported {len(exported)} != rows {data_norm.shape[0]}",
        }

    rng = random.Random(int(args.seed) + 17)
    by_rat: dict[str, list[int]] = {}
    for i, row in exported.iterrows():
        by_rat.setdefault(str(row["rat_id"]), []).append(int(i))
    n_swapped = 0
    for idxs in by_rat.values():
        if len(idxs) < 2:
            continue
        order = list(range(len(idxs)))
        rng.shuffle(order)
        src = data_norm[idxs].copy()
        src_raw = data_raw[idxs].copy()
        data_norm[idxs] = src[order]
        data_raw[idxs] = src_raw[order]
        n_swapped += sum(1 for a, b in enumerate(order) if a != b)

    group_data = build_group_data(
        data_norm,
        n_chunks=8,
        feature_selection=feature_selection,
        data_raw=data_raw,
        n_points_per_day=int(args.n_points),
        fixed_n_days=8,
    )
    scores = _fit_group_data(
        group_data,
        args=args,
        feature_selection=feature_selection,
        parameter_selection=parameter_selection,
    )
    result = {
        "status": "ok",
        "null_mode": "row_shuffle_within_rat_cached_proxy",
        "note": (
            "Calendar rebuild unavailable; permuted cached profile rows within rat. "
            "Not a full shuffle_dates rebuild — treat as smoke null."
        ),
        "n_row_positions_moved": n_swapped,
        "feature": args.feature,
        "n_points": args.n_points,
        "overlap": args.overlap,
        "range_likelihood": args.range_likelihood,
        "mean_likelihood": args.mean_likelihood,
        "tune": args.tune,
        "draws": args.draws,
        "chains": args.chains,
        "seed": args.seed,
        **scores,
    }
    out_path = out_dir / f"mcmc_result_seed{args.seed}_{args.mode}_rowshuffle_proxy.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[neg-ctrl] Wrote proxy null {out_path}")
    return result


def _run_mcmc_optional(args: argparse.Namespace, shuffled: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Best-effort single-config fit; returns status dict (may be skipped)."""
    try:
        from seismic_pipeline.config.changepoint_defaults import default_export_base_cfg
        from seismic_pipeline.features.rem_chunk_features import build_group_data, prepare_model_data
        from seismic_pipeline.features.runtime import set_runtime_export_cfg
        from seismic_pipeline.seismo.rem_export import export_rem_profiles_10days_cached_only
    except ImportError as exc:
        return {"status": "skipped", "reason": f"import failed: {exc}"}

    feature_selection = _feature_selection(args.feature)
    parameter_selection = _parameter_selection(
        args.feature,
        mean_lik=args.mean_likelihood,
        range_lik=args.range_likelihood,
    )

    export_dir = out_dir / f"profiles_shuffled_seed{args.seed}_{args.mode}"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_cfg = dict(default_export_base_cfg(output_dir=str(export_dir)))
    export_cfg["events"] = [dict(x) for x in shuffled]
    export_cfg["n_points_per_day"] = int(args.n_points)
    export_cfg["overlap"] = float(args.overlap)
    export_cfg["window_days"] = int(args.tau_upper)  # 8-day convention with tau_upper=8
    export_cfg["output_dir"] = str(export_dir)
    # Prefer standalone hypnogram cache used by July runs
    hyp_cache = PROJECT_ROOT / "hypnogram_cache_legacy10"
    if hyp_cache.is_dir():
        export_cfg["local_hypnogram_cache_dir"] = str(hyp_cache)
    set_runtime_export_cfg(export_cfg)

    print(
        "[neg-ctrl] Exporting REM profiles for shuffled events "
        f"(feature={args.feature}, N={args.n_points}, ov={args.overlap})…"
    )
    try:
        export_result = export_rem_profiles_10days_cached_only(**export_cfg)
        nanpad = export_result["paths"]["nanpad_output_csv"]
        prep = prepare_model_data(csv_path=nanpad)
        group_data = build_group_data(
            prep["data_norm"],
            n_chunks=int(args.tau_upper),
            feature_selection=feature_selection,
            data_raw=prep["data_raw"],
            n_points_per_day=int(args.n_points),
            fixed_n_days=int(args.tau_upper),
        )
        scores = _fit_group_data(
            group_data,
            args=args,
            feature_selection=feature_selection,
            parameter_selection=parameter_selection,
        )
        result = {
            "status": "ok",
            "null_mode": "shuffle_dates_rebuild",
            "feature": args.feature,
            "n_points": args.n_points,
            "overlap": args.overlap,
            "range_likelihood": args.range_likelihood,
            "mean_likelihood": args.mean_likelihood,
            "tune": args.tune,
            "draws": args.draws,
            "chains": args.chains,
            "seed": args.seed,
            "export_nanpad": nanpad,
            **scores,
        }
        out_path = out_dir / f"mcmc_result_seed{args.seed}_{args.mode}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[neg-ctrl] Wrote {out_path}")
        return result
    except Exception as exc:  # noqa: BLE001 — surface data/S3 issues; fall back to proxy
        print(f"[neg-ctrl] Calendar rebuild failed: {exc}", flush=True)
        print("[neg-ctrl] Falling back to within-rat row-shuffle proxy on cached profiles…", flush=True)
        try:
            proxy = _row_shuffle_within_rat_null(
                args, out_dir, feature_selection, parameter_selection
            )
            proxy["calendar_rebuild_error"] = str(exc)
            return proxy
        except Exception as exc2:  # noqa: BLE001
            return {
                "status": "skipped",
                "reason": f"feature build failed (need local profiles/S3): {exc}; proxy also failed: {exc2}",
            }

def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    events_all = _load_default_events()
    events = [dict(x) for x in events_all]
    if args.before_only:
        events = [e for e in events if e.get("direction") == "before"]

    if args.mode == "shuffle_dates":
        shuffled = shuffle_dates_within_rat(events, seed=args.seed)
    else:
        shuffled = permute_event_labels(events, seed=args.seed)

    summary = summarize_shuffle(events, shuffled)
    prior_mean = 0.5 * (args.tau_lower + args.tau_upper)

    payload = {
        "mode": args.mode,
        "seed": args.seed,
        "before_only": bool(args.before_only),
        "n_original": len(events_all),
        "n_used": len(events),
        "summary": summary,
        "tau_prior": {
            "dist": "DiscreteUniform",
            "lower": args.tau_lower,
            "upper": args.tau_upper,
            "prior_mean_under_flat": prior_mean,
        },
        "success_criteria": SUCCESS_CRITERIA,
        "primary_recommendation": {
            "feature": "daily:mean+range",
            "n_points": 24,
            "overlap": 0.5,
            "range_likelihood": "beta_constrained",
            "note": "Do not use plain beta as primary (boundary artifact).",
        },
        "events_original": events,
        "events_shuffled": shuffled,
        "checklist": "reports/research_integrity_checklist.md §5",
    }

    events_path = out_dir / f"events_{args.mode}_seed{args.seed}.json"
    events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Negative control (integrity) ===")
    print(f"mode={args.mode}  seed={args.seed}  before_only={args.before_only}")
    print(
        f"dates changed: {summary['n_date_changed']}/{summary['n_events']} "
        f"({summary['frac_date_changed']:.0%})"
    )
    print(f"Wrote {events_path}")
    print()
    print(SUCCESS_CRITERIA)
    print()
    print(
        "CLI for full MCMC (expensive):\n"
        "  python scripts/run_negative_control_shuffle.py --no-dry-run --run-mcmc "
        f"--mode {args.mode} --seed {args.seed} "
        "--feature daily:mean+range --n-points 24 --overlap 0.5 "
        "--range-likelihood beta_constrained"
    )

    # Cheap smoke: shuffle must move at least one date for full 8-day list
    if summary["n_events"] >= 2 and summary["n_date_changed"] == 0 and args.mode == "shuffle_dates":
        # Possible if each rat has a single event — warn, do not fail
        print("[warn] No dates changed; check per-rat event counts.")
    elif args.mode == "shuffle_dates" and summary["n_date_changed"] == 0:
        print("[fail] shuffle_dates changed 0 dates unexpectedly.", file=sys.stderr)
        return 1

    # Determinism smoke
    again = (
        shuffle_dates_within_rat(events, seed=args.seed)
        if args.mode == "shuffle_dates"
        else permute_event_labels(events, seed=args.seed)
    )
    if again != shuffled:
        print("[fail] Non-deterministic shuffle for fixed seed.", file=sys.stderr)
        return 1
    print("[ok] Dry-run smoke: deterministic shuffle + artifact written.")

    if args.run_mcmc:
        mcmc = _run_mcmc_optional(args, shuffled, out_dir)
        mcmc_path = out_dir / f"mcmc_status_seed{args.seed}_{args.mode}.json"
        mcmc_path.write_text(json.dumps(mcmc, indent=2), encoding="utf-8")
        print(f"[neg-ctrl] MCMC status: {mcmc.get('status')} → {mcmc_path}")
        if mcmc.get("status") == "ok":
            e_tau = mcmc.get("e_tau", mcmc.get("tau_mean"))
            print(
                f"  null_mode={mcmc.get('null_mode')}  "
                f"E[τ]={float(e_tau):.3f}  "
                f"HDI60w={mcmc.get('tau_hdi_60_width')}  "
                f"elpd/feat·evt={mcmc.get('elpd_loo_per_feature_event'):.3f}  "
                f"(compare to real primary; prior mean≈{prior_mean:.1f})"
            )
        else:
            print(f"  skipped/failed: {mcmc.get('reason')}")
            print("  Dry-run artifacts remain valid; re-run when profiles/S3 are available.")
    else:
        print("[ok] Skipped MCMC (default). Pass --run-mcmc when ready for a full null fit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
