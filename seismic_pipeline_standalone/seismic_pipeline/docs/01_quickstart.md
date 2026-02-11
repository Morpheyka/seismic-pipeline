# Quickstart

This guide shows how to **assemble a working pipeline** using the public `seismic_pipeline` API.

## 1) Data model (what `X` and `y` are)

### Input events (`X`)

Most experiments start from a list of events:

```python
events = [
  {"rat_id": "R2", "date": "2022-11-07"},
  {"rat_id": "R5", "date": "2024-02-19"},
]
```

- `rat_id`: string like `"R2"`.
- `date`: event date. Both `YYYY-MM-DD` and `YYYY_MM_DD` appear in this codebase; many components try both.

### Targets (`y`)

Many pipelines **do not provide y upfront**. Instead, `EventLabelGeneratorYt` (or `CustomEventLabelGeneratorYt`) expands each event into one or more windows and generates labels (0/1).

## 2) Target-aware pipeline basics (`PipelineYt`)

The defining contract:

- Each transformer in the pipeline returns `(X_out, y_out)`.
- `PipelineYt.fit(X, y=None)` propagates `(X, y)` through steps, then fits the final estimator on the transformed `(X, y)`.
- `PipelineYt.predict(X)` returns `(y_pred, y_true_transformed)` so scoring can use the labels aligned with the final features.

## 3) Minimal end-to-end pipeline

```python
from sklearn.linear_model import LogisticRegression

from seismic_pipeline import (
    PipelineYt,
    EventLabelGeneratorYt,
    REMProfileCalculatorYt,
    REMProfileCleanerYt,
    REMProfileMaxMinExtractorYt,
    HypnogramCacheManagerYt,
)

events = [
    {"rat_id": "R2", "date": "2022-11-07"},
    {"rat_id": "R3", "date": "2022-12-01"},
]

cache = HypnogramCacheManagerYt(
    local_cache_dir="./hypnogram_cache",
    # local_data_root defaults to /home/ponomattik/mnt/wd/rat in this standalone repo
    # s3_config can be provided to enable S3 fallback
)

pipe = PipelineYt([
    ("label_generator", EventLabelGeneratorYt(window_days=3)),
    ("rem_calc", REMProfileCalculatorYt(cache_manager=cache, fail_on_missing_data=False)),
    ("clean", REMProfileCleanerYt(replacement_method="median")),
    ("extract", REMProfileMaxMinExtractorYt(include_other_stats=True)),
    ("classifier", LogisticRegression(max_iter=200)),
])

pipe.fit(events)  # y is generated internally
y_pred, y_true = pipe.predict(events)
```

### Notes

- `REMProfileCalculatorYt` may **filter out samples** if a hypnogram cannot be loaded; it returns a filtered `y` that matches the filtered `X`.
- If `fail_on_missing_data=True`, `REMProfileCalculatorYt` raises when any hypnogram is missing. This is useful for grid search to “reject” invalid parameter combos.

## 4) Hyperparameter search (`GridSearchCVYt`)

`GridSearchCVYt` is a target-aware wrapper around scikit-learn’s `GridSearchCV`. It works best with `PipelineYt` and a scorer that understands `(y_pred, y_true_transformed)`.

```python
from sklearn.linear_model import LogisticRegression
from seismic_pipeline import PipelineYt, GridSearchCVYt
from seismic_pipeline.mod.scoreryt import yt_accuracy_scorer
from seismic_pipeline import EventLabelGeneratorYt, REMProfileCalculatorYt, HypnogramCacheManagerYt

cache = HypnogramCacheManagerYt(local_cache_dir="./hypnogram_cache")

pipe = PipelineYt([
    ("label_generator", EventLabelGeneratorYt(window_days=3)),
    ("rem_calc", REMProfileCalculatorYt(cache_manager=cache, fail_on_missing_data=False)),
    ("classifier", LogisticRegression(max_iter=200)),
])

param_grid = {
    "label_generator__window_days": [3, 5, 7],
    "rem_calc__window_size_hours": [4, 6],
    "classifier__C": [0.1, 1.0, 10.0],
}

gs = GridSearchCVYt(pipe, param_grid=param_grid, scoring=yt_accuracy_scorer, n_jobs=4, verbose=2)
gs.fit(events)
```

## 5) Visualization & reports

Visualization tools are available via `seismic_pipeline.visualization`:

```python
from seismic_pipeline.visualization import visualize_hyperparameter_grid_slices, ReportGenerator

visualize_hyperparameter_grid_slices(gs, output_dir="./results")

report = ReportGenerator()
report.head_title("Experiment overview")
report.head_best_params(gs.best_params_)
report.head_scores({"best_cv_score": float(gs.best_score_)})
report.create_json("./results/reports", "experiment.json", write="rewrite")
report.compile("./results/reports", "./results/run_report", "experiment.json")
```



