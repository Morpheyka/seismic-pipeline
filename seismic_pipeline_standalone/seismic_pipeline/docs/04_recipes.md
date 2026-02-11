# Recipes

Practical “how do I…” tasks for the seismic pipeline.

## Change event windowing strategy

### Standard (two-window) labeling

Use `EventLabelGeneratorYt(window_days=N)`:

- Positive window: `N` days immediately before event (label 1)
- Control window: the `N` days before the positive window (label 0)

### Multi-window labeling

Use `CustomEventLabelGeneratorYt` with `window_step_days` and `window_mode` to control window positions and generate multiple positive/control windows per event.

Tip: Always verify the expansion factor, because it controls class balance and CV size.

## Work with missing hypnograms (local + S3 fallback)

`REMProfileCalculatorYt` attempts:

1. load from cache
2. cache from **local source**
3. if local fails, optionally check/download from **S3 temp bucket**

Behavior control:

- `REMProfileCalculatorYt(..., fail_on_missing_data=True)` (default): raise if anything is missing (useful for grid search).
- `fail_on_missing_data=False`: skip missing samples (and filter `y` accordingly).

Cache settings:

```python
from seismic_pipeline import HypnogramCacheManagerYt

cache = HypnogramCacheManagerYt(
  local_cache_dir="./hypnogram_cache",
  local_data_root="/home/ponomattik/mnt/wd/rat",
  s3_config=None,  # set dict to enable S3
  s3_temp_bucket="temp",
)
```

## Swap feature extraction: daily vs summary features

### Daily per-day statistic

```python
from seismic_pipeline import REMDailyExtractorYt
daily = REMDailyExtractorYt(daily_statistic="mean", window_days=6)
```

### Multiple daily stats

```python
from seismic_pipeline import REMDailyMultiStatExtractorYt
multi = REMDailyMultiStatExtractorYt(daily_statistics=["max_min_diff", "mean"], window_days=6)
```

### Summary stats (max/min and more)

```python
from seismic_pipeline import REMProfileMaxMinExtractorYt
summary = REMProfileMaxMinExtractorYt(include_other_stats=True)
```

## Debug intermediate datasets

Use `save_step_data` to write `X`/`y` to CSV between pipeline steps.

```python
from seismic_pipeline import save_step_data

X_df, y_df = save_step_data(X, y, step_name="rem_calc", output_dir="./debug", output_prefix="exp1", step_number=2)
```

## Run a grid search and produce a report

```python
from seismic_pipeline import GridSearchCVYt
from seismic_pipeline.mod.scoreryt import yt_accuracy_scorer
from seismic_pipeline.visualization import visualize_hyperparameter_grid_slices, ReportGenerator

gs = GridSearchCVYt(pipe, param_grid=param_grid, scoring=yt_accuracy_scorer, n_jobs=8, verbose=2)
gs.fit(events)

visualize_hyperparameter_grid_slices(gs, output_dir="./results")

report = ReportGenerator()
report.head_title("Experiment overview")
report.head_best_params(gs.best_params_)
report.head_scores({"best_cv_score": float(gs.best_score_)})
report.create_json("./results/reports", "experiment.json", write="rewrite")
report.compile("./results/reports", "./results/run_report", "experiment.json")
```


