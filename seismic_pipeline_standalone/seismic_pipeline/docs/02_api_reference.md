# API reference (public surface)

This page documents the primary objects intended for import from:

```python
import seismic_pipeline as sp
```

The authoritative public exports live in `seismic_pipeline/__init__.py`.

## Target-aware core (`seismic_pipeline.mod`)

### `TransformerMixinYt`

Base mixin for target-aware transformers.

- **Purpose**: ensures transformers preserve and return `y` alongside `X`.
- **Key behavior**: default `fit_transform` calls `fit(...).transform(X, y)` and expects `(X_out, y_out)`.

Source: `seismic_pipeline/mod/sklearnbaseyt.py`

### `PipelineYt`

Target-aware replacement for `sklearn.pipeline.Pipeline`.

- **fit**: propagates `(X, y)` through steps; final estimator is fit on transformed `(X, y)`.
- **predict**: returns `(y_pred, y_true_transformed)` (tuple), not just `y_pred`.
- **transform**: returns `(X_out, y_out)` after applying all steps.

Source: `seismic_pipeline/mod/pipelineyt.py`

### `GridSearchCVYt`

Target-aware `GridSearchCV`.

- **Default CV**: if `cv` is not provided, uses `StratifiedKFoldYt` for classifiers, `KFoldYt` otherwise.
- **Scoring**: designed to work with a scorer like `yt_accuracy_scorer` when `PipelineYt.predict` returns tuples.
- **Extra**: `calculate_best_params_metrics(...)` can recompute multiple metrics for the best parameters.

Source: `seismic_pipeline/mod/grid_searchyt.py`

### `yt_accuracy_scorer`

Scorer for target-aware pipelines that return `(y_pred, y_true_transformed)` from `predict`.

- Returns `0.0` on empty inputs, mismatch sizes, or exceptions (keeps grid search robust).

Source: `seismic_pipeline/mod/scoreryt.py`

### `save_step_data`

Debugging helper to dump intermediate `X`/`y` to CSV during pipeline execution.

Source: `seismic_pipeline/mod/utilsyt.py`

## Seismic components (`seismic_pipeline.seismo`)

### `EventLabelGeneratorYt`

Expands each event into two windows and generates binary labels:

- **label 1**: window immediately before the event
- **label 0**: window preceding the positive window

Key parameters:

- `window_days: int = 3`
- `random_seed: Optional[int] = None`
- `date_format: str = "%Y_%m_%d"`

Input `X`: list of `{"rat_id": str, "date": str}`

Output `X`: list of dicts with `window_dates` + metadata fields; output `y`: `np.ndarray` of 0/1.

Source: `seismic_pipeline/seismo/event_transformers.py`

### `CustomEventLabelGeneratorYt`

Custom windowing strategy for event labeling (used in “range/steps” experiments).

Source: `seismic_pipeline/seismo/event_transformers.py`

### `HypnogramCacheManagerYt`

Cache manager for hypnogram files with support for:

- local/network data root
- optional S3 download (rat bucket + temp bucket)
- a local cache index (`cache_index.pkl`) and date-format migration

Key parameters:

- `local_cache_dir: str = "./hypnogram_cache"`
- `local_data_root: str = "/home/ponomattik/mnt/wd/rat"` (standalone default)
- `s3_config: Optional[Dict] = None`
- `s3_rat_bucket: str = "rat"`
- `s3_temp_bucket: str = "temp"`
- `target_channels: List[str] = ["cxf","cxb","htl","hcm"]`
- `sampling_rate: int = 250`

Source: `seismic_pipeline/seismo/hypnogram_cache_manager.py`

### `REMProfileCalculatorYt`

For each input row with a `rat_id` and `window_dates`, calculates REM profiles per day and concatenates them.

Key parameters:

- `cache_manager: Optional[HypnogramCacheManagerYt]`
- `window_size_hours: int = 6`
- `step_size_hours: int = 1`
- `rem_stage: int = 2`
- `epoch_length_sec: int = 5`
- `sampling_rate: int = 250`
- `profile_features: List[str] = ["rem_percentage"]`
- `fail_on_missing_data: bool = True` (raise if any hypnogram is missing)

Source: `seismic_pipeline/seismo/rem_profile_calculator.py`

### `REMProfileCombinerYt`

Combines multiple profiles (e.g., multiple channels / derived features) into a single feature vector.

Source: `seismic_pipeline/seismo/rem_profile_calculator.py`

### `REMProfileCleanerYt` / `REMProfileAdvancedCleanerYt`

Cleans REM profiles by replacing zeros/NaNs and handling profiles with insufficient valid data.

Key parameters (`REMProfileCleanerYt`):

- `replace_zeros: bool = True`
- `replace_nans: bool = True`
- `replacement_method: str = "median"` (`median|mean|mode|constant`)
- `min_valid_fraction: float = 0.5`
- `handle_empty_profiles: str = "skip"` (`skip|zero|nan`)

Source: `seismic_pipeline/seismo/rem_profile_cleaner.py`

### `REMDailyExtractorYt`

Extracts one statistic per day from a REM profile window (e.g. `max_min_diff`, `mean`, etc.).

Key parameters:

- `daily_statistic: str = "max_min_diff"`
- `window_days: int = 3`
- `handle_empty_days: str = "zero"`

Source: `seismic_pipeline/seismo/rem_daily_extractor.py`

### `REMDailyMultiStatExtractorYt`

Extracts **multiple** per-day statistics (multi-feature daily vector).

Key parameters:

- `daily_statistics: List[str] = ["max_min_diff", "mean"]`
- `window_days: int = 3`
- `handle_empty_days: str = "zero"`

Source: `seismic_pipeline/seismo/rem_daily_extractor.py`

### `REMProfileMaxMinExtractorYt`

Extracts summary statistics from each profile (default: `max` and `min`, optional extras).

Key parameters:

- `include_other_stats: bool = False`
- `stats_to_extract: Optional[List[str]] = None`
- `handle_empty_profiles: str = "zero"`

Source: `seismic_pipeline/seismo/rem_maxmin_extractor.py`

### `REMProfileSummaryExtractorYt`

Extracts a richer set of summary statistics including temporal + distribution characteristics.

Source: `seismic_pipeline/seismo/rem_maxmin_extractor.py`

### `MetadataAdderYt`

Optionally appends numeric encodings of metadata to the feature matrix.

- Default metadata columns: `["original_event_date", "original_rat_id"]`

Source: `seismic_pipeline/seismo/metadata_adder.py`

## Visualization (`seismic_pipeline.visualization`)

### `visualize_hyperparameter_grid_slices(...)`

Creates 1D/2D/3D+ slice visualizations of a grid search result.

Source: `seismic_pipeline/visualization/hyperparameter_grid_visualizer.py`

### `ReportGenerator`

Builds experiment reports (JSON sections → compiled markdown, optional PDF).

Source: `seismic_pipeline/visualization/report_generator.py`


