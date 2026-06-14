# Exhaustive Search Implementation Summary

## 1) What is implemented as "exhaustive search" in this project

In this project, exhaustive search is implemented as a complete enumeration of changepoint-model configurations, followed by Bayesian fitting and unified scoring for each configuration.  
The main runtime entry point (as used in `otchet_V_0.2_8day.ipynb`) is:

- `rlib._generate_exhaustive_configs(EXHAUSTIVE_OPTIONS)`
- `rlib.exhaustive_model_search(...)`
- `rlib.summarize_exhaustive_search(...)`
- `rlib.export_exhaustive_search_results_to_csv(...)`
- `rlib.plot_exhaustive_search_results(...)`

So, the method is not a heuristic optimizer: it traverses the full combinatorial model space defined by the proposal options.

---

## 2) Main libraries used

From `seismic_pipeline_standalone/rem_profiles_export_10days_lib.py` imports:

- **Core numerics/data**
  - `numpy` (`np`)
  - `pandas` (`pd`)
- **Bayesian modeling and diagnostics**
  - `pymc` (`pm`) for model definition and MCMC sampling
  - `arviz` (`az`) for WAIC/LOO, BFMI, and posterior diagnostics
  - `pytensor.tensor` (`pt`) for tensor ops used in the marginalized-`tau` model
- **Visualization**
  - `matplotlib.pyplot` (`plt`) for exhaustive-search diagnostic plots
- **Utilities**
  - `itertools` for Cartesian products/combinations
  - `json` for stable config keys and exports
  - `hashlib` for compact config fingerprints
  - `copy`, `math`, `time`, `warnings`, `os`
- **Optional progress / accelerated sampling**
  - `tqdm.auto.tqdm` (if available) for progress bar
  - JAX backends through PyMC (`nuts_backend="numpyro"` or `"blackjax"`)
- **Project-local data pipeline**
  - `seismic_pipeline.HypnogramCacheManagerYt`
  - `seismic_pipeline.REMProfileCalculatorYt`

---

## 3) How parameters are realized in code

### 3.1 Proposal-grid parameters (search space definition)

`EXHAUSTIVE_OPTIONS` in notebook defines:

- `rem_profile_choices`: list of REM profile parameter triplets
  - `window_size_hours`
  - `step_size_hours`
  - `rem_stage`
- `n_chunks_choices`: possible `n_chunks` values for temporal feature splitting
- `allowed_groups`: subset of `{all, odd, even, concat}`
- `allowed_metrics`: subset of `{mean, range, std, skewness, kurtosis}` (actual allowed set validated against defaults)
- `likelihood_choices_by_metric`: per-metric likelihood family options
- `tau_threshold`: threshold used in `P(tau > threshold)` metrics
- optional `pareto_threshold`: threshold for Pareto-k filtering during LOO robustness checks

### 3.2 Exhaustive configuration generation

`_generate_exhaustive_configs(...)` does:

1. validates all proposal-option lists are non-empty;
2. normalizes each REM profile tuple with `_normalize_rem_profile_params(...)`;
3. forms all non-empty subsets of `(group, metric)` blocks;
4. for active metrics, builds Cartesian products over allowed likelihood families;
5. creates final config dict with:
   - `rem_profile_params`
   - `n_chunks`
   - `feature_selection`
   - `parameter_selection`
   - `tau_threshold`

### 3.3 Model-level parameters (fitting and inference)

In `exhaustive_model_search(...)`, key runtime parameters are:

- MCMC:
  - `draws`, `tune`
  - `chains`, `cores`
  - `nuts_backend` in `{"pymc", "numpyro", "blackjax"}`
- changepoint structure:
  - `tau_mode` in `{"discrete", "marginalized"}`
  - `tau_lower`, `tau_upper`
- robustness:
  - `cache_fits`
  - `seed`
  - `pareto_threshold` (from proposal options)
- control:
  - `verbose`
  - `progressbar`

### 3.4 Feature realization and normalization terms

- `build_group_data(...)` converts `feature_selection` into data blocks by group (`all/odd/even/concat`) and metric, with strict check that all selected blocks share the same chunk count (required by shared `tau`).
- `n_active_features` is computed as:
  - `sum(len(metrics) for metrics in config["feature_selection"].values())`
- Per-result normalization fields:
  - `elpd_loo_per_event = elpd_loo / E'`
  - `elpd_loo_per_feature = elpd_loo / (F * E')`
  - where `F = n_active_features`, and `E' = n_model_events` (after possible Pareto-k removals).

### 3.5 Why per-event normalization is mostly optional in the main search

Before Pareto-k corrective drops, models in one run use the same base event count `E` (same `data_work_base` for all configs), so ranking by raw `elpd_loo` is already comparable by event count at that stage.  
The implementation still computes normalized scores to remain robust when `E'` changes after Pareto-k retries and when feature counts differ across configs.

---

## 4) Core flow of the exhaustive pipeline

1. **Grid generation**  
   `_generate_exhaustive_configs(...)` builds all candidate configurations.
2. **Deduplication by model semantics**  
   `_exhaustive_model_signature(...)` removes degenerate duplicates.
3. **Per-config model build and sampling**
   - `build_group_data(...)`
   - `build_changepoint_model(...)`
   - `sample_model(...)`
4. **Scoring and diagnostics**
   - `score_changepoint_trace(..., criterion="loo", loo_report="elpd")`
   - diagnostics: `r_hat_max`, `ess_min_bulk`, `ess_min_tail`, `n_divergences`, `bfmi`
5. **Pareto-k robustness loop for top influence points**
   - detect high Pareto-k via `_collect_pareto_k_stats(...)`
   - if needed, drop worst event index and refit/re-score
6. **Ranking and outputs**
   - rank by `elpd_loo_per_feature` (descending)
   - aggregate with `summarize_exhaustive_search(...)`
   - export CSV/JSON and plots

---

## 5) Function extracts (signatures + key listings)

### 5.1 `_generate_exhaustive_configs`

```python
def _generate_exhaustive_configs(
    proposal_options: dict,
    *,
    tau_threshold: float | None = None,
) -> list[dict]:
    """Generate all valid exhaustive-search configurations from proposal grid."""
```

Critical listing (subset/product logic):

```python
block_space = sorted((g, m) for g in allowed_groups for m in allowed_metrics)
for subset in _nonempty_subsets(block_space):
    ...
    for likelihood_combo in itertools.product(*metric_likelihood_choices):
        out.append({...})
```

### 5.2 `exhaustive_model_search`

```python
def exhaustive_model_search(
    proposal_options: dict,
    data_norm: np.ndarray,
    *,
    draws: int = 500,
    tune: int = 1000,
    nuts_backend: str = "pymc",
    ...
) -> dict:
    """Evaluate all changepoint model configurations from a proposal grid."""
```

Critical listing (fit + score + normalization):

```python
group_data = build_group_data(data_work, n_chunks=int(config["n_chunks"]), feature_selection=config["feature_selection"])
model = build_changepoint_model(group_data, tau_lower=int(tau_lower), tau_upper=tu, parameter_selection=config["parameter_selection"], tau_mode=tau_mode)
trace = sample_model(model, draws=draws, tune=tune, nuts_backend=nuts_backend, chains=chains, cores=cores, progressbar=False)
score_parts = score_changepoint_trace(trace, group_data=group_data, parameter_selection=config["parameter_selection"], model=model, criterion="loo", loo_report="elpd")
```

```python
n_active_features = int(sum(len(metrics or []) for metrics in (config.get("feature_selection") or {}).values()))
record["elpd_loo_per_event"] = float(record["elpd_loo"]) / float(n_model_events)
record["elpd_loo_per_feature"] = float(record["elpd_loo"]) / float(n_active_features * n_model_events)
```

### 5.3 `build_changepoint_model`

```python
def build_changepoint_model(
    group_data: dict,
    tau_lower: int = 2,
    tau_upper: int | None = None,
    parameter_selection: dict | None = None,
    tau_mode: str = "discrete",
):
    """Build PyMC changepoint model for selected feature blocks."""
```

Critical listing (marginalized `tau` realization):

```python
if tau_mode == "marginalized":
    log_w = loglik_by_tau - np.log(float(n_tau))
    pointwise_log_lik = pm.math.logsumexp(log_w_rows, axis=0)
    pm.Deterministic("changepoint_pointwise_log_lik", pointwise_log_lik)
    pm.Potential("tau_marginalized_logp", pm.math.logsumexp(log_w))
```

### 5.4 `sample_model`

```python
def sample_model(
    model,
    draws: int = 4000,
    tune: int = 2000,
    *,
    nuts_backend: str = "pymc",
    chains: int = 4,
    cores: int | None = None,
    progressbar: bool = True,
):
    """Run MCMC sampling using PyMC or JAX-based NUTS backend."""
```

Critical listing (backend switch):

```python
if backend == "pymc":
    trace = pm.sample(**sample_kwargs)
elif backend in {"numpyro", "blackjax"}:
    trace = pm.sample(nuts_sampler=backend, nuts_sampler_kwargs=nuts_sampler_kwargs, **sample_kwargs)
```

### 5.5 `score_changepoint_trace`

```python
def score_changepoint_trace(
    trace,
    *,
    group_data: dict,
    parameter_selection: dict | None,
    tau_threshold: float = 7.0,
    model=None,
    criterion: str = "waic",
    loo_report: str = "ic",
) -> dict[str, Any]:
    """Compute tau/mixing diagnostics and WAIC/LOO-based comparison statistics."""
```

Critical listing (LOO scale selection):

```python
ic_loo = az.loo(idata_ic, scale="log")
elpd_loo = _float_ic_scalar(getattr(ic_loo, "elpd_loo", float("nan")))
loo_ic = _float_ic_scalar(getattr(ic_loo, "loo", float("nan")))
loo_stat = elpd_loo if str(loo_report).strip().lower() == "elpd" else loo_ic
```

### 5.6 `summarize_exhaustive_search`

```python
def summarize_exhaustive_search(search_result: dict) -> dict:
    """Build summary stats and top-model views from exhaustive run output."""
```

Critical listing (valid-model filter):

```python
valid = [
    r for r in results
    if r.get("status") == "ok"
    and float(r.get("r_hat_max", float("inf"))) <= 1.05
    and float(r.get("ess_min_bulk", float("-inf"))) >= 100.0
    and int(r.get("n_divergences", 1)) == 0
]
```

---

## 6) Practical output artifacts used in report writing

From notebook call chain, exhaustive search produces:

- `run_output_8day/exhaustive_search_results.csv`  
  full per-config records with scores, diagnostics, and serialized config.
- `run_output_8day/exhaustive_search_summary.json`  
  compact aggregate summary (`n_total`, `n_fitted`, `n_valid`, best fingerprints, etc.).
- diagnostic plots via `plot_exhaustive_search_results(...)`.

These artifacts are suitable as primary tables/figures for the exhaustive-search chapter.
