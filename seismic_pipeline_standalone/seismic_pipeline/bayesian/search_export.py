"""CSV/JSON export for exhaustive changepoint model search results."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from seismic_pipeline.bayesian.search import (
    _config_feature_label,
    summarize_exhaustive_search,
)


def _config_likelihood_label(config: dict) -> str:
    ps = config.get("parameter_selection") or {}
    parts = []
    for metric_name in sorted(ps):
        likelihood = (ps[metric_name] or {}).get("likelihood", "")
        parts.append(f"{metric_name}={likelihood}")
    return "; ".join(parts)


def export_exhaustive_search_results_to_csv(
    search_result: dict,
    output_csv: str,
    *,
    output_summary_json: str | None = None,
) -> dict[str, str]:
    """Export exhaustive-search model records to CSV and optional summary JSON."""
    csv_dir = os.path.dirname(output_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for rank, rec in enumerate(search_result.get("results") or [], start=1):
        cfg = rec.get("config") or {}
        rem = cfg.get("rem_profile_params") or {}
        rows.append(
            {
                "rank_by_loo": rank,
                "fingerprint": rec.get("fingerprint"),
                "status": rec.get("status"),
                "error": rec.get("error"),
                "features": _config_feature_label(cfg),
                "likelihoods": _config_likelihood_label(cfg),
                "n_chunks": cfg.get("n_chunks"),
                "tau_threshold": cfg.get("tau_threshold"),
                "window_size_hours": rem.get("window_size_hours"),
                "step_size_hours": rem.get("step_size_hours"),
                "rem_stage": rem.get("rem_stage"),
                "loo": rec.get("loo"),
                "elpd_loo": rec.get("elpd_loo", rec.get("loo")),
                "elpd_loo_per_event": rec.get("elpd_loo_per_event"),
                "elpd_loo_per_feature": rec.get("elpd_loo_per_feature"),
                "elpd_loo_per_feature_event": rec.get("elpd_loo_per_feature_event"),
                "waic": rec.get("waic"),
                "r_hat_max": rec.get("r_hat_max"),
                "ess_min_bulk": rec.get("ess_min_bulk"),
                "ess_min_tail": rec.get("ess_min_tail"),
                "bfmi": rec.get("bfmi"),
                "n_divergences": rec.get("n_divergences"),
                "tau_map": rec.get("tau_map"),
                "tau_map_concentration": rec.get("tau_map_concentration"),
                "p_tau_gt_threshold": rec.get("p_tau_gt_threshold"),
                "e_tau": rec.get("e_tau"),
                "tau_std": rec.get("tau_std"),
                "p_tau_gt_6": rec.get("p_tau_gt_6"),
                "loo_pareto_k_max": rec.get("loo_pareto_k_max"),
                "loo_n_over_threshold": rec.get("loo_n_over_threshold"),
                "waic_warning_flag": rec.get("waic_warning_flag"),
                "n_feature_blocks": rec.get("n_feature_blocks"),
                "n_active_features": rec.get("n_active_features"),
                "elapsed_time": rec.get("elapsed_time"),
                "data_shape": str(rec.get("data_shape")),
                "data_output_dir": rec.get("data_output_dir"),
                "bad_sample_indices": str(rec.get("bad_sample_indices")),
                "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            }
        )

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_csv, index=False)

    paths: dict[str, str] = {"results_csv": output_csv}
    if output_summary_json:
        summary_dir = os.path.dirname(output_summary_json)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary_payload: dict[str, Any] = {
            **summarize_exhaustive_search(search_result),
            "elapsed_total_sec": float(search_result.get("elapsed_total", float("nan"))),
            "results_csv": output_csv,
            "n_rows_saved": int(len(df_results)),
        }

        def _json_default(obj: Any) -> Any:
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(output_summary_json, "w", encoding="utf-8") as f:
            json.dump(
                summary_payload,
                f,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        paths["summary_json"] = output_summary_json
    return paths
