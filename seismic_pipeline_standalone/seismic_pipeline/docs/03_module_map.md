# Module map (how pieces compose)

This page is intended for “agent-style” use: given a goal, quickly identify which module/class to use and how it plugs into a pipeline.

## Dataflow overview

Typical dataflow:

1. **Events** (`X`: list of `{rat_id, date}`)  
2. **Label generation** → expands events into windows + produces `y`  
3. **Hypnogram access (cache manager)** → local and optional S3 fallback  
4. **REM profile calculation** → per-window numeric vectors  
5. **Cleaning / feature extraction** → 2D feature matrix  
6. **Classifier + CV / grid search**  
7. **Reports + visualizations**

## Components table

| Layer | Class / function | Module | Input `X` | Output `X` | Notes |
|---|---|---|---|---|---|
| Pipeline core | `PipelineYt` | `mod/pipelineyt.py` | any | any | Propagates `(X, y)` through steps; `predict` returns `(y_pred, y_true_transformed)` |
| Base mixin | `TransformerMixinYt` | `mod/sklearnbaseyt.py` | any | any | Base for target-aware transformers |
| Search | `GridSearchCVYt` | `mod/grid_searchyt.py` | any | any | Default target-aware CV; works with `yt_accuracy_scorer` |
| Scoring | `yt_accuracy_scorer` | `mod/scoreryt.py` | any | float | Handles `(y_pred, y_true)` tuple return |
| Debug | `save_step_data` | `mod/utilsyt.py` | any | — | Writes intermediate datasets to CSV |
| Event labeling | `EventLabelGeneratorYt` | `seismo/event_transformers.py` | list[dict] | list[dict] | Creates `window_dates` and produces `y` |
| Event labeling (custom) | `CustomEventLabelGeneratorYt` | `seismo/event_transformers.py` | list[dict] | list[dict] | Used for “steps-range” style experiments |
| Hypnogram cache | `HypnogramCacheManagerYt` | `seismo/hypnogram_cache_manager.py` | — | — | Loads/caches hypnograms; can try local then S3 temp |
| REM profiles | `REMProfileCalculatorYt` | `seismo/rem_profile_calculator.py` | list[dict] | `np.ndarray` | May filter rows with missing data; can raise when missing |
| Cleaning | `REMProfileCleanerYt` | `seismo/rem_profile_cleaner.py` | list[np.ndarray] | list[np.ndarray] | Replaces zeros/NaNs, can skip low-quality profiles |
| Daily features | `REMDailyExtractorYt` | `seismo/rem_daily_extractor.py` | list/array | `np.ndarray` | One statistic per day |
| Daily multi-features | `REMDailyMultiStatExtractorYt` | `seismo/rem_daily_extractor.py` | list/array | `np.ndarray` | Multiple statistics per day |
| Summary features | `REMProfileMaxMinExtractorYt` | `seismo/rem_maxmin_extractor.py` | list/array | `np.ndarray` | Max/min (+ optional extra stats) |
| Summary features | `REMProfileSummaryExtractorYt` | `seismo/rem_maxmin_extractor.py` | list/array | `np.ndarray` | Richer summary set |
| Metadata | `MetadataAdderYt` | `seismo/metadata_adder.py` | `np.ndarray` | `np.ndarray` | Appends numeric metadata columns (optional) |
| Viz | `visualize_hyperparameter_grid_slices` | `visualization/hyperparameter_grid_visualizer.py` | grid-search | images | Produces slice heatmaps / bar charts |
| Reporting | `ReportGenerator` | `visualization/report_generator.py` | artifacts | md/json | JSON sections → compiled markdown (optional PDF) |

## Changepoint / Bayesian stack (8-day notebooks)

Separate from the sklearn classifier pipeline; optional deps in `requirements-bayesian.txt`.

| Layer | Module | Role |
|---|---|---|
| Report orchestration | `changepoint_report.py` (standalone) | CSV load, diagnostics, MCMC reruns, t-tests, `build_changepoint_report()` |
| Defaults | `config/changepoint_defaults.py` | `DEFAULT_EVENTS_10D`, `FEATURE_SELECTION_PRESETS`, env-based S3 config |
| REM export | `seismo/rem_export.py` | `export_rem_profiles_10days_cached_only` from hypnogram cache |
| Chunk features | `features/rem_chunk_features.py` | `build_group_data`, `prepare_model_data`, max-min scaling |
| Runtime | `features/runtime.py` | `ChangepointRunContext`, shared export/prepare cfg |
| PyMC model | `bayesian/changepoint_model.py` | `build_changepoint_model`, `sample_model` |
| Scoring | `bayesian/diagnostics.py` | WAIC/LOO, `collect_pareto_k_stats`, trace summaries |
| Search | `bayesian/search.py` | `exhaustive_model_search`, Hamming distance |
| MH search | `bayesian/mh_search.py` | `metropolis_hastings_model_search` |
| Runner | `bayesian/runner.py` | `run_variant`, posterior predictive checks |
| Plots | `visualization/changepoint_plots.py` | `plot_exhaustive_search_results`, trace plots |
| Facade | `rem_profiles_export_10days_lib.py` | Backward-compatible re-exports for notebooks |

Dataflow for changepoint studies:

1. **Events** → `export_rem_profiles_10days_cached_only` → REM CSV matrices  
2. **prepare_model_data** → normalized matrix + sample mask  
3. **build_group_data** → chunk feature DataFrames per group  
4. **build_changepoint_model** → PyMC model → **sample_model** / **run_variant**  
5. **changepoint_report** → diagnostics, reruns, ReportGenerator markdown

## Recommended “agent heuristics”

If you’re building an agent that assembles pipelines from docs, these heuristics work well:

- Prefer pipelines that start with an **event label generator** (so `y` is produced).
- After label generation, ensure you pick exactly one path that produces a **2D numeric feature matrix** before the classifier.
- If a transformer can drop samples (missing hypnograms, low-valid-fraction profiles), treat it as a **filter** that also filters `y`.
- For grid search, prefer scorers that explicitly support tuple-return `predict` (e.g. `yt_accuracy_scorer`).



