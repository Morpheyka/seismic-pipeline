# Seismo preprocessors used by `otchet_V_0.1.ipynb`

## 1) What is actually used in the notebook

The notebook does **not** import `seismo` classes directly.  
Instead, it imports `rem_profiles_export_10days_lib.py`, and that module uses `seismo` internals (`HypnogramCacheManagerYt`, `REMProfileCalculatorYt`) under the hood.

Primary notebook path:

1. Add `seismic_pipeline_standalone` to `sys.path`.
2. Import helpers from `rem_profiles_export_10days_lib`.
3. Define events (`EVENTS_10D`) and export config (`export_cfg` + `rem_profile_params`).
4. Call `export_rem_profiles_10days_cached_only(...)` to build REM-profile matrices from cached hypnograms.
5. Run `prepare_model_data(...)`, `set_runtime_data_norm(...)`.
6. Run scenarios via `run_variant(...)`; when `rem_profile_params` changes, this can trigger REM-profile recomputation.

---

## 2) Code listing: notebook entry points

From `otchet_V_0.1.ipynb` (code cell content):

```python
# Make seismic_pipeline package importable from notebook.
WORKSPACE_ROOT = os.getcwd()
SEISMIC_STANDALONE_DIR = os.path.join(WORKSPACE_ROOT, "seismic_pipeline_standalone")
if SEISMIC_STANDALONE_DIR not in sys.path:
    sys.path.insert(0, SEISMIC_STANDALONE_DIR)

from rem_profiles_export_10days_lib import (
    export_rem_profiles_10days_cached_only,
    compute_chunk_features,
)
```

```python
rem_profile_params = {
    "window_size_hours": 6,
    "step_size_hours": 1,
    "rem_stage": 2,
}

export_cfg = {
    "events": EVENTS_10D,
    "output_csv_nanpad": "samples_10days_nanpad.csv",
    "local_data_root": "/home/ponomattik/mnt/wd/rat",
    "local_hypnogram_cache_dir": "./hypnogram_cache",
    "epoch_length_sec": 5,
    "sampling_rate": 250,
    "concat_hypnogram_for_event": False,
    "require_full_window_for_concat": True,
    "drop_incomplete_events": True,
}

export_result = export_rem_profiles_10days_cached_only(**export_cfg, **rem_profile_params)
```

```python
trace_20_mixed, summary_20_mixed = run_variant(
    n_chunks=20,
    feature_selection=feature_selection_mixed,
    parameter_selection=parameter_selection_mixed,
    rem_profile_params=rem_profile_params,
    draws=4000,
    tune=8000,
    tau_lower=2,
    tau_upper=10,
)
```

---

## 3) Which preprocessors/dataset handlers from `seismo` matter here

Even though the notebook works via one high-level library, the effective preprocessing chain is in `seismo`:

### 3.1 `HypnogramCacheManagerYt`
- File: `seismic_pipeline_standalone/seismic_pipeline/seismo/hypnogram_cache_manager.py`
- Role:
  - resolve a `(rat_id, date)` hypnogram from local cache, local source, or S3;
  - keep cache index and a negative-cache for known missing pairs.
- Why it matters:
  - all downstream REM features depend on successful hypnogram retrieval.

### 3.2 `REMProfileCalculatorYt` (main feature generator)
- File: `seismic_pipeline_standalone/seismic_pipeline/seismo/rem_profile_calculator.py`
- Role:
  - for each `(rat_id, window_dates)` sample, compute daily REM-profile vectors and concatenate;
  - or provide per-day vectors that `rem_profiles_export_10days_lib.py` then packs/pads.
- Why it matters:
  - contains `_fraction(...)`, the exact hypnogram -> feature formula.

### 3.3 `REMDailyExtractorYt`, `REMDailyMultiStatExtractorYt`, `REMDailyMultiDynamicStatExtractorYt`
- File: `seismic_pipeline_standalone/seismic_pipeline/seismo/rem_daily_extractor.py`
- Role:
  - post-process long REM profile vectors into per-day scalar summaries (`mean`, `range`, `std`, etc.).
- Why it matters:
  - these are downstream aggregators; they do **not** replace the core REM% computation.

### 3.4 `REMProfileMaxMinExtractorYt`, `REMProfileSummaryExtractorYt`
- File: `seismic_pipeline_standalone/seismic_pipeline/seismo/rem_maxmin_extractor.py`
- Role:
  - global statistical summarization of already built REM profile vectors.
- Why it matters:
  - useful for dimensionality reduction, but still secondary to `_fraction(...)`.

---

## 4) How the notebook triggers the `seismo` pipeline (indirectly)

The call chain is:

`otchet_V_0.1.ipynb`  
-> `export_rem_profiles_10days_cached_only(...)` in `rem_profiles_export_10days_lib.py`  
-> `_build_10day_inputs(...)` (expand each event into 10 dates)  
-> `_cache_needed_dates(...)` (load/cache hypnograms per day)  
-> `REMProfileCalculatorYt` (or concat mode path)  
-> `REMProfileCalculatorYt._fraction(...)` (compute REM percentages)  
-> pad vectors and write CSV/metadata.

Important listing from export library:

```python
from seismic_pipeline import HypnogramCacheManagerYt, REMProfileCalculatorYt

cache_manager = HypnogramCacheManagerYt(...)
rows = _build_10day_inputs(events)
cached_pairs, missing_pairs = _cache_needed_dates(cache_manager, rows)

rem_calc = REMProfileCalculatorYt(
    cache_manager=cache_manager,
    window_size_hours=window_size_hours,
    step_size_hours=step_size_hours,
    rem_stage=rem_stage,
    epoch_length_sec=epoch_length_sec,
    sampling_rate=sampling_rate,
    fail_on_missing_data=False,
)

if concat_hypnogram_for_event:
    raw_features, valid_indices = _build_concat_features(...)
else:
    raw_features, valid_indices = rem_calc._calculate_features_for_X(export_rows)
```

---

## 5) Main focus: how features are calculated from hypnograms

### 5.1 Input representation assumptions
- Hypnogram is an epoch-wise stage sequence (often stored as list where first element is actual array).
- `rem_stage` (default `2`) marks REM epochs.
- `epoch_length_sec` (default `5`) defines epoch duration.

Extraction logic:

```python
if isinstance(hypnogram, list):
    if len(hypnogram) > 0:
        hypno_data = hypnogram[0]
...
rem_profile = self._fraction(
    hypno_data,
    (self.window_size_hours, self.step_size_hours),
    self.rem_stage,
    self.epoch_length_sec,
)
```

### 5.2 Exact `_fraction(...)` formula

From `REMProfileCalculatorYt._fraction`:

```python
points_hour = int(3600 / window_sec)
start = 0
end = points_hour * slide_hours[0]

while end < len(hypno):
    stage_count = len(np.where(hypno[start:end] == stage)[0])
    percentage = 100 * stage_count / (end - start)
    res.append(percentage)
    start += points_hour * slide_hours[1]
    end += points_hour * slide_hours[1]
```

This means:

- `points_hour = 3600 / epoch_length_sec`
- `window_points = points_hour * window_size_hours`
- `step_points = points_hour * step_size_hours`
- for each sliding window:
  - `REM% = 100 * count(stage == rem_stage in window) / window_length`

So, features are **percentages of REM-labeled epochs per time window**, not spectral features directly.

### 5.3 Example with defaults
- `epoch_length_sec=5` -> `points_hour=720`
- `window_size_hours=6` -> `window_points=4320`
- `step_size_hours=1` -> `step_points=720`

Each output point is REM percentage over a 6-hour epoch-window, shifted every 1 hour.

---

## 6) Relationship to hypnogram generation

In this notebook path, hypnograms are expected to be already available in cache/local/S3 (`cached_only` export).  
However, stage semantics come from hypnogram generation logic in `seismo/hypnogram_calculator.py`:

```python
part_hypno[(mask2 == 1)] = 2
...
hypno = no_singles(hypno, 0)
hypno = no_singles(hypno, 1)
hypno = no_rems_in_wake(hypno, 2)
```

And frequency features used when hypnograms are generated:

```python
def delta_theta(eps, func, dband=(2.5, 5.5), tband=(5.5, 8)):
    ...
    ratios = theta_power / deltas
```

So the complete conceptual chain is:
EEG -> staged hypnogram (including REM label `2`) -> sliding-window REM%.

---

## 7) Two export modes and their implications

In `rem_profiles_export_10days_lib.py` there are two ways to derive event vectors:

1. **Default (`concat_hypnogram_for_event=False`)**
   - Compute day-by-day REM profile and concatenate day vectors.
2. **Concat-hypnogram mode (`concat_hypnogram_for_event=True`)**
   - Concatenate raw hypnograms across all window days first;
   - then run one `_fraction(...)` over the long sequence.

Concat mode listing:

```python
big_hypno = np.concatenate(hypno_parts)
rem_profile = rem_calc._fraction(
    big_hypno,
    (rem_calc.window_size_hours, rem_calc.step_size_hours),
    rem_calc.rem_stage,
    rem_calc.epoch_length_sec,
)
```

These modes can produce different vectors near day boundaries.

---

## 8) Important caveats and edge cases

1. **Tail dropped:** `_fraction` uses `while end < len(hypno)`; incomplete trailing window is ignored.
2. **Missing hypnograms:** missing `(rat_id, date)` are tracked by negative cache; events can be dropped via `drop_incomplete_events`.
3. **Invalid/no-data stage values:** if hypnogram has non-REM labels only (or invalid labels), REM% becomes low/zero but denominator still uses full window length.
4. **List hypnogram format:** only first element of list-like hypnogram is used.
5. **Daily extractor split heuristic:** `samples_per_day = len(profile) // window_days` in daily extractor is approximate if day lengths differ or profile has irregular structure.
6. **`sampling_rate` in REMProfileCalculatorYt:** stored parameter, but `_fraction` discretization is driven by `epoch_length_sec`.

---

## 9) Short summary

- Notebook-level preprocessing control is in `rem_profile_params` and export config.
- Real feature math is in `REMProfileCalculatorYt._fraction`.
- Core hypnogram-derived feature is sliding-window **REM percentage**:
  - `100 * (# REM epochs / # epochs in window)`.
- Other `seismo` REM transformers are mostly aggregators/summarizers over already computed REM profiles.
