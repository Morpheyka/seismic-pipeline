"""Bayesian changepoint scenario runner."""
from __future__ import annotations

import numpy as np
import pandas as pd

from seismic_pipeline.bayesian.changepoint_model import build_changepoint_model, sample_model
from seismic_pipeline.bayesian.diagnostics import (
    _available_varnames,
    _sampler_stat,
    feature_likelihood_profiles,
    summary_from_trace,
    tau_probabilities,
)
from seismic_pipeline.features.rem_chunk_features import (
    build_group_data,
    prepare_model_data,
    set_runtime_data_norm,
)
from seismic_pipeline.features.runtime import (
    _RUNTIME_DATA_NORM,
    get_runtime_data_raw,
    get_runtime_export_cfg,
    get_runtime_prepare_cfg,
)
from seismic_pipeline.seismo.rem_export import (
    _normalize_rem_profile_params,
    export_rem_profiles_10days_cached_only,
)
from seismic_pipeline.visualization.changepoint_plots import (
    plot_posteriors_like_script,
    plot_trace_and_tau,
)
from seismic_pipeline.visualization.changepoint_ppc import plot_posterior_predictive_check

__all__ = ["run_variant", "plot_posterior_predictive_check"]


def run_variant(
    title: str,
    *,
    n_chunks: int,
    feature_selection,
    parameter_selection: dict | None = None,
    rem_profile_params: dict | None = None,
    draws: int = 4000,
    tune: int = 2000,
    nuts_backend: str = "pymc",
    chains: int = 4,
    cores: int | None = None,
    progressbar: bool | None = None,
    tau_mode: str = "discrete",
    tau_lower: int = 2,
    tau_upper: int | None = None,
    data_norm: np.ndarray | None = None,
    data_raw: np.ndarray | None = None,
    day_lengths_per_sample: list[list[int]] | None = None,
    csv_path: str | None = None,
    plot_likelihood_profiles: bool = True,
    likelihood_profile_grid_size: int = 300,
    likelihood_profile_log_density_y: bool | set[tuple[str, str]] | None = None,
    return_likelihood_profiles: bool = False,
):
    """Full scenario run: features -> model -> MCMC -> summary -> plots."""
    normalized_rem_profile_params: dict[str, int] | None = None
    if rem_profile_params is not None:
        required_keys = {"window_size_hours", "step_size_hours", "rem_stage"}
        missing = sorted(required_keys - set(rem_profile_params))
        if missing:
            raise ValueError(
                "rem_profile_params is missing required keys: "
                f"{missing}. Expected keys: {sorted(required_keys)}"
            )
        w, s, r = _normalize_rem_profile_params(
            window_size_hours=rem_profile_params["window_size_hours"],
            step_size_hours=rem_profile_params["step_size_hours"],
            rem_stage=rem_profile_params["rem_stage"],
        )
        normalized_rem_profile_params = {
            "window_size_hours": w,
            "step_size_hours": s,
            "rem_stage": r,
        }

    backend = str(nuts_backend).strip().lower()
    if backend in {"numpyro", "blackjax"} and str(tau_mode).strip().lower() == "discrete":
        print(
            "JAX NUTS backend with discrete tau is unsupported; using tau_mode='marginalized'."
        )
        tau_mode = "marginalized"

    data_for_run = data_norm if data_norm is not None else _RUNTIME_DATA_NORM
    data_raw_for_run = data_raw if data_raw is not None else get_runtime_data_raw()
    prep_cfg = get_runtime_prepare_cfg() or {}
    if day_lengths_per_sample is None:
        day_lengths_per_sample = prep_cfg.get("day_lengths_per_sample")
    if csv_path is None:
        csv_path = prep_cfg.get("csv_path")
    window_days = None
    export_cfg = get_runtime_export_cfg()
    if export_cfg is not None:
        window_days = int(export_cfg.get("window_days", 10))
    if normalized_rem_profile_params is not None and data_norm is None:
        if export_cfg is None:
            print(
                "rem_profile_params were provided, but no prior export config is available. "
                "Using existing runtime data without recalculation."
            )
        else:
            export_cfg = dict(export_cfg)
            export_cfg.update(normalized_rem_profile_params)
            print("Recalculating REM profiles with updated rem_profile_params...")
            export_result = export_rem_profiles_10days_cached_only(**export_cfg)

            prep_cfg = get_runtime_prepare_cfg() or {}
            prep_csv_path = export_result["paths"]["nanpad_output_csv"]
            prep_bad_indices = prep_cfg.get("bad_sample_indices", None)
            prep = prepare_model_data(
                csv_path=prep_csv_path,
                bad_sample_indices=prep_bad_indices,
            )
            set_runtime_data_norm(prep["data_norm"])
            data_for_run = prep["data_norm"]
            data_raw_for_run = prep["data_raw"]
            day_lengths_per_sample = prep.get("day_lengths_per_sample")
            csv_path = prep.get("csv_path", prep_csv_path)
            if get_runtime_export_cfg() is not None:
                window_days = int(get_runtime_export_cfg().get("window_days", 10))
            print(
                "Recalculation complete: "
                f"data_norm shape={data_for_run.shape}, csv_path={prep_csv_path!r}"
            )

    if data_for_run is None:
        raise ValueError("data_norm is not set. Pass data_norm=... or call set_runtime_data_norm(...).")

    print("=" * 90)
    print(f"Сценарий: {title}")
    print(f"  n_chunks={n_chunks}")
    print(f"  feature_selection={feature_selection!r}")
    if parameter_selection is not None:
        print(f"  parameter_selection={parameter_selection!r}")
    if normalized_rem_profile_params is not None:
        print(f"  rem_profile_params={normalized_rem_profile_params!r}")
        print(
            "  note: rem_profile_params trigger recalculation only when runtime export "
            "and prepare configs are available (or when data_norm is passed explicitly)."
        )
    print(f"  nuts_backend={nuts_backend!r}")
    print(f"  tau_mode={tau_mode!r}")
    print("=" * 90)

    group_data = build_group_data(
        data_for_run,
        n_chunks=n_chunks,
        feature_selection=feature_selection,
        data_raw=data_raw_for_run,
        window_days=window_days,
        day_lengths_per_sample=day_lengths_per_sample,
        csv_path=csv_path,
    )
    for group_name, features in group_data.items():
        for feat_name, df in features.items():
            print(f"Группа '{group_name}', признак '{feat_name}': форма {df.shape}")
            print(f"Первые 2 строки ({feat_name}):")
            print(df.head(2).to_string(index=False))
            print()

    model = build_changepoint_model(
        group_data,
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        parameter_selection=parameter_selection,
        tau_mode=tau_mode,
    )
    trace = sample_model(
        model,
        draws=draws,
        tune=tune,
        nuts_backend=nuts_backend,
        chains=chains,
        cores=cores,
        progressbar=(
            False
            if progressbar is None and str(nuts_backend).strip().lower() in {"numpyro", "blackjax"}
            else True if progressbar is None else bool(progressbar)
        ),
    )
    available_vars = _available_varnames(trace)
    trace_vars = []
    summary_vars = []
    if "tau" in available_vars:
        trace_vars.append("tau")
        summary_vars.append("tau")
    if "tau_mean" in available_vars:
        trace_vars.append("tau_mean")
        summary_vars.append("tau_mean")

    for group_name, features in group_data.items():
        for feat_name in features.keys():
            for param_name in ("mu", "sigma", "alpha", "beta"):
                p1 = f"{param_name}_{group_name}_{feat_name}_1"
                p2 = f"{param_name}_{group_name}_{feat_name}_2"
                if p1 in available_vars and p2 in available_vars:
                    trace_vars.extend([p1, p2])
                    summary_vars.extend([p1, p2])
            nu_name = f"nu_{group_name}_{feat_name}"
            if nu_name in available_vars:
                summary_vars.append(nu_name)

    if summary_vars:
        summary = summary_from_trace(trace, summary_vars)
        print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())
    else:
        summary = pd.DataFrame()
        print("Summary: no scalar parameters selected for compact table.")

    diverging = _sampler_stat(trace, "diverging")
    n_div = int(np.asarray(diverging).sum())
    print(f"Дивергенции: {n_div}")

    energy = np.asarray(_sampler_stat(trace, "energy"), dtype=float).reshape(-1)
    if energy.size > 1 and np.var(energy) > 0:
        bfmi = float(np.mean(np.diff(energy) ** 2) / np.var(energy))
        print(f"BFMI (approx): {bfmi:.3f}")
    else:
        print("BFMI (approx): недостаточно данных.")

    support, probs = tau_probabilities(trace)
    print("Вероятности tau:")
    for k, p in zip(support, probs):
        print(f"  P(tau={k}) = {p:.3f}")
    map_idx = int(np.argmax(probs))
    print(f"MAP tau: {int(support[map_idx])}, концентрация: {float(probs[map_idx]):.3f}")

    plot_trace_and_tau(trace, trace_vars, title_prefix=title)
    plot_posteriors_like_script(trace, group_data=group_data, title_prefix=title)
    likelihood_profiles = feature_likelihood_profiles(
        trace,
        group_data=group_data,
        parameter_selection=parameter_selection,
        grid_size=likelihood_profile_grid_size,
        plot=plot_likelihood_profiles,
        log_density_y=likelihood_profile_log_density_y,
    )
    if return_likelihood_profiles:
        return trace, summary, likelihood_profiles
    return trace, summary
