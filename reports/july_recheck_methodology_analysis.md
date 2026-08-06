# July 2026 recheck: n_points / overlap methodology — analysis and recommendations

**Date:** 2026-08-06  
**Scope:** re-analysis of July rewritten changepoint pipeline (fixed windows → `n_points_per_day` + `overlap`), comparison with the 896-config W/S study, and synthesis with older neuroseismo manuscripts.

---

## 1. Executive summary

The July recheck on rewritten code **does not overturn** the core scientific claim of the thesis/preprint: a shared REM changepoint remains near **\(\mathbb{E}[\tau]\approx 6.5\)–\(6.7\) days** before the event. What *does* change under the new profile parameterization and feature grouping is **which features win LOO ranking**.

| Study | Grid | Top feature class | \(\mathbb{E}[\tau]\) (best quartile / top-50) |
|-------|------|-------------------|-----------------------------------------------|
| 896-config (thesis) | \(W\in\{2,3,6\}\), \(S\in\{1,2\}\) | **`concat:mean`** (+ student_t / lognormal) | \(\approx 6.64\)–\(6.73\) |
| July parallel full | \(N\in\{12,24,48\}\), overlap \(\in\{0,0.25,0.5\}\) | **`daily:range`** (+ beta); then daily mean+range | \(\approx 6.55\) (top-50), \(6.70\) (Q1) |
| July shape_shift-only | same \(N\times\)overlap | odd/even `shape_shift` (gamma) | \(\approx 5.15\) (weaker, often near prior edge) |

**Bottom line for research design:** keep the Bayesian \(\tau\) conclusion; **re-centre features on daily range (and daily mean+range)**; treat **`shape_shift` as secondary/exploratory**; keep **student_t / normal + beta** as primary likelihoods; plan **skew-normal / zero-inflated beta** as the next prior–likelihood upgrade. Profiles around **2024-09-30 (R2/R3)** and several incomplete **after_reversed** windows are the main poorly detectable / high-Pareto cases.

---

## 2. Methodology shift: fixed W/S → n_points + overlap

### 2.1 Old parameterization (896-config / thesis)

REM daily profiles were built from sliding windows:

- `window_size_hours` \(W \in \{2,3,6\}\)
- `step_size_hours` \(S \in \{1,2\}\)
- `rem_stage = 2`

Feature blocks: `concat` / `even` / `odd` × `{mean, range}` with shared chunked \(\tau\).

### 2.2 New parameterization (July rewritten pipeline)

Canonical profile params (see `config/changepoint_defaults.py`, `docs/changepoint_exhaustive_config_report.md`):

| Parameter | Grid | Bounds |
|-----------|------|--------|
| `n_points_per_day` | `{12, 24, 48}` | `[4, 96]` |
| `overlap` | `{0.0, 0.25, 0.5}` | `[0, 1)` |
| `rem_stage` | `2` | — |

Legacy `(W,S)` is still accepted but **deprecated** and converted:

\[
n_{\text{points}} = 24/S,\qquad
\text{overlap} = 1 - S/W \quad (W > S).
\]

| Old \((W,S)\) | Equivalent \((n,\text{overlap})\) |
|---------------|-------------------------------------|
| (2, 1) — thesis favourite | (24, 0.50) |
| (2, 2) | (12, 0.00) |
| (3, 1) | (24, 0.67) — **outside** new default grid |
| (6, 1) | (24, 0.83) — **outside** new default grid |

So the July grid is **not a strict superset** of the old W/S grid: it drops high-overlap 3–6 h windows, but **adds** finer resolution \(N=48\) and an explicit non-overlapping / partial-overlap axis.

### 2.3 Why the rewrite matters scientifically

1. **Identifiability:** \(N\) and overlap separate “how many daily samples” from “how much temporal smoothing,” instead of entangling them in \((W,S)\).
2. **Comparability with classical neuroseismo work:** older Saevskiy et al. manuscripts used **12 non-overlapping 2 h bins** \(\equiv (N=12,\ \text{overlap}=0)\).
3. **New feature ontology in the main July run:** `daily` / `day` / `night` aggregates (one scalar per day or half-day) rather than only concat/even/odd chunk series — closer to the classical “daily range of the REM profile” pipeline.

Code anchors:

- `seismic_pipeline_standalone/seismic_pipeline/config/changepoint_defaults.py` (`REM_PROFILE_CHOICES`, `validate_rem_profile_params`)
- `seismic_pipeline_standalone/seismic_pipeline/seismo/rem_export.py`
- `seismic_pipeline_standalone/seismic_pipeline/features/rem_chunk_features.py`
- Entry points: `scripts/run_parallel_search.py`, `run_parallel_search_8day_events.py`, `run_parallel_search_shape_shift.py`

---

## 3. Evidence from July rechecks (concrete artifacts)

### 3.1 Main July exhaustive search (`run_output_8day_parallel_full`)

**Artifact:** `seismic_pipeline_standalone/run_output_8day_parallel_full/exhaustive_search_parallel.csv`

| Quantity | Value |
|----------|-------|
| Configs completed | **1224** (all `status=ok`) |
| Rank-eligible | **1079** |
| REM grid | \(N\in\{12,24,48\}\times\) overlap \(\in\{0,0.25,0.5\}\) |
| Event set | 36 listed windows; **33 exported** complete 8-day profiles |
| Scoring | PSIS-LOO ELPD; rank by `elpd_loo_per_feature` |

**Top-15 eligible (abridged):**

| Rank | Features | Likelihoods | \(N\) | ov | ELPD/feat | \(\mathbb{E}[\tau]\) |
|------|----------|-------------|------|-----|-----------|----------------------|
| 1 | daily: range | range=beta | 48 | 0.00 | 6.63 | 6.28 |
| 2 | daily: range | range=beta | 48 | 0.50 | 6.24 | 7.98 |
| 3 | daily: range | range=beta | 48 | 0.25 | 6.08 | 5.57 |
| 4 | daily: mean, range | student_t + beta | 48 | 0.00 | 6.03 | 6.47 |
| 5 | daily: mean, range | normal + beta | 48 | 0.00 | 5.99 | 6.80 |
| 6 | daily: range | range=beta | 24 | 0.00 | 5.82 | 5.70 |

**Top-50 structure:**

- Features: `daily: mean, range` (28), `day: range` (15), `daily: range` (14), `daily: mean` (8); night appears but is secondary.
- Likelihoods: `range=beta` (34), `mean=normal` (22), `mean=student_t` (17), `range=interval_inflated_beta` (11).
- Resolution: **\(N=48,\ \text{ov}=0\)** dominates (26/50); then (48, 0.25), (48, 0.50). Coarse \(N=12\) is almost absent from the top.
- \(\mathbb{E}[\tau]\) top-50: **mean 6.55, median 6.62**; eligible Q1 (best ELPD quartile): **6.70 ± 0.93**.

**Longer MCMC refits** (`top10_refits/`, tune=6000, draws=3000) confirm the same ranking core. Example posterior means for \(\tau\):

| Refit rank | Config | \(\tau\) mean (refit) |
|------------|--------|------------------------|
| 1 | daily:range, beta, N=48, ov=0 | 6.28 |
| 2 | daily:range, beta, N=48, ov=0.5 | 7.98 (near upper bound — caution) |
| 4 | daily:mean+range, student_t+beta, N=48, ov=0 | 6.52 |
| 5 | daily:mean+range, normal+beta, N=48, ov=0 | 6.78 |

Source: `…/top10_refits/refit_summary.json`, `…/rank*/posterior_summary.csv`.

### 3.2 Shape-shift ablation (`run_output_8day_parallel_shape_shift`)

**Artifact:** `run_output_8day_parallel_shape_shift/exhaustive_search_parallel.csv` (126 configs).

- Best models: **`odd:shape_shift` / even+odd**, likelihood **gamma**, mostly **\(N=12\)**.
- ELPD per feature is **negative** (\(\approx -6.4\) best) — not competitive with daily range/mean models (positive ELPD/feat \(\approx 5\)–\(6.6\)).
- \(\mathbb{E}[\tau]\) top-20 \(\approx 5.15\) with frequent MAP at the lower edge (3) — **weak / unstable changepoint signal**.
- **Every** completed run removes event index **5 = R2 2024-09-30** (Pareto retry) — systematic outlier for shape features.

Conclusion: shape/L1–L2 style dynamics are interesting descriptively (see §3.3) but **should not be primary** for the shared-\(\tau\) Bayesian search.

### 3.3 REM profile form study (13 Jul 2026)

**Artifact:** `reports/rem_profile_artifact_cleaned_study.tex`

- Protocol already uses **\(N=24\), overlap=0.50** (and checks all 9 \(N\times\)overlap cells).
- Descriptive late-vs-early L1/L2 rise is **stable across all 9 profile grids**.
- Formal tests (days 6–8 vs 4–5; within-block distances) are mostly **non-significant**; one exploratory half-profile L1 signal (\(p_{\text{one}}=0.0449\)).
- Explicit artifacts excluded: row 0 day 1; rows 4–5 day 3; row 22 day 4.
- Author caveat: half-profile ≠ verified photoperiod until mapping is checked.

### 3.4 Likelihood / prior empirical diagnostics

**Artifacts:**

- `seismic_pipeline_standalone/run_output_likelihood_fit_diagnostics/likelihood_flexibility_recommendation.json`
- `seismic_pipeline_standalone/run_output_8day_parallel_full/likelihood_recommendation.json`
- summarized in `docs/changepoint_exhaustive_config_report.md` §6

Key empirical mismatches with the *current* exhaustive grid:

| Metric | Observed shape | Grid issue | Prefer |
|--------|----------------|------------|--------|
| mean (concat/even/odd, day-norm) | platykurtic, mild skew | student_t/normal underfit; lognormal wrong support | skewnorm, gennorm, 2-Gaussian mixture |
| mean (daily) | near-symmetric | OK | normal ≈ student_t |
| range (concat) | ~35% mass near 0 | IIB@0.9 hits *upper* bound | **zero-inflated beta** / left inflation |
| range (daily) | some mass near 1 | IIB appropriate | beta + IIB |
| shape_shift | right skew | already OK | **gamma** ≳ lognormal |
| std | right skew from 0 | lognormal loses large ΔAIC | gengamma / exponweib |

Operational recommendation already used in the July full search (`likelihood_recommendation.json`): mean ∈ `{normal, student_t}`; range ∈ `{beta, interval_inflated_beta}` with `support_upper=2` and `/2` scaling.

### 3.5 Old vs rewritten code benchmark (engineering, not science)

**Artifacts:** `run_output_old_vs_current_exhaustive/benchmark_summary_final.json`, `selected_configs_from_june.json`

- Selected June winners were still **W/S `concat:mean`** configs.
- Current blackjax runs were slower and not numerically comparable 1:1 (different observation builder / ELPD scale) — treat as **implementation stress test**, not a scientific contradiction.

### 3.6 Predictive risk line (context only)

`reports/seismic_risk_predictive_study.tex` + `run_output_risk_model/`: dataset built (54 rows), **CV not finished**. Orthogonal to the changepoint recheck but part of the July workstream.

---

## 4. Insights from `literature/old_neuroseismoreports`

Three Word manuscripts (pre-Bayesian / parallel neuroseismo line):

1. **`manuscript_REM_quakes.docx`** — “Изменение характеристик парадоксального сна…”  
2. **`manuscript_REM_forecast.docx`** — short-term forecast rule from REM changes  
3. **`Био_сейсмопрогнозирование_25_09_2025_САИ.docx`** — station-level bio-seismic monitoring summary  

### 4.1 Feature doctrine in the classical line

- Daily REM **profile** from **non-overlapping 2 h windows → 12 points/day** (\(\equiv N=12,\ \text{overlap}=0\)).
- Primary scalar: **range (размах)** of that profile — difference max−min, intentionally mixing light/dark extrema.
- Profiles min–max scaled per event neighbourhood before pooling.

This **aligns with the July Bayesian winner (`daily:range`)** more closely than with the thesis-favoured `concat:mean`.

### 4.2 Timing claims

- Significant range anomalies **up to ~2–4 days before** and **~2–4 days after** strong events (intensity >2).
- Strongest post-event distortion often on days **+1…+3**; pre-event effect clearest on **−2, −1** in effect-size plots.
- Forecast manuscript: threshold rule on REM changes over **~3 days** pre-event; expanded event set (12 vs 6).

### 4.3 Agreement / tension with Bayesian \(\tau\approx 6.6\)

| Classical neuroseismo | Bayesian changepoint (896 + July) |
|-----------------------|-----------------------------------|
| Effect visible 2–4 d pre-event | Changepoint **~6.5–6.7 d** pre-event |
| Emphasizes **range** | Thesis emphasized **concat:mean**; July recheck restores **daily:range** |
| Non-overlapping 2 h bins (N=12) | July prefers **N=48** for daily aggregates; N=12 still classical-compatible |
| Includes strong **post-event** reaction | Models use `before` and `after_reversed` windows; after windows contribute many Pareto outliers |

**Interpretation (justified, not proven):** classical tests detect when the *effect becomes large relative to quiet controls*; the Bayesian single-\(\tau\) model places the *onset of a new regime* earlier. These are compatible if the deviation grows over several days. They are **not** the same estimand.

---

## 5. Priors worth keeping (justified)

Defaults live in `bayesian/priors.py` (`_default_parameter_selection`, IIB helpers) and `PARAMETER_SELECTION_PRESETS`.

### 5.1 Keep as primary research priors

| Target | Prior / likelihood | Why |
|--------|--------------------|-----|
| Shared \(\tau\) | Discrete uniform on \(\{3,\ldots,8\}\) (marginalized) | Matches both studies; posterior mass stays interior for good models; thesis Q1 \(\approx 6.6\) |
| **daily / concat mean** location | \(\mu\sim\mathrm{Normal}(0,1.5)\), \(\sigma\sim\mathrm{HalfNormal}(1)\); lik. **student_t** or **normal**; \(\nu\sim\mathrm{Exp}(0.05)+2\) | July top ranks; daily mean near-symmetric; student_t still useful under contamination |
| **daily range** | Beta likelihood on range/2; \(\alpha,\beta\sim\mathrm{Gamma}(\mu=3,\sigma=1.5)\) (code default for beta) | Dominates July LOO; matches classical “размах”; bounded support |
| **IIB for daily range** | \(\pi\sim\mathrm{Beta}(1,10)\), \(\alpha,\beta\sim\mathrm{Gamma}(\mu=3,\sigma=1)\), threshold \(0.9\) | Competitive in top-50 when daily range has upper saturation; keep as *secondary* |
| **shape_shift** (exploratory) | \(\mu\sim\mathrm{Normal}(-2,1)\), \(\sigma\sim\mathrm{HalfNormal}(0.5)\); lik. **gamma** | Empirical gamma ≳ lognormal; priors match small positive shifts |
| g-prior on \(\mu\) | **`none`** for exhaustive ranking | Avoids confounding model comparison; use `hyper_g_n` / Zellner–Siow only in sensitivity checks |

### 5.2 Deprioritize or redesign

| Spec | Action | Why |
|------|--------|-----|
| **lognormal for day-normalized mean** | Demote from primary grid | Wrong support / poor MLE fit; July grid already dropped it for the daily run |
| **lognormal for range** | Avoid as primary | Bounded [0, 2]; beta family wins |
| **IIB@0.9 for concat range** | Do not treat as default | Mass is near **0**, not 1 — need **zero-inflated / left-boundary** inflation |
| Very wide kurtosis/skewness priors | Keep only if those metrics re-enter the grid | Not in winning July feature sets |

### 5.3 Next prior–likelihood upgrades (implementation priority)

From MLE diagnostics (July):

1. **Skew-normal** (mean, day/night mean)  
2. **Zero-inflated beta** (concat/even/odd range)  
3. **gennorm / 2-Gaussian mixture** (platykurtic concat mean)  
4. Optional **Kumaraswamy** for interior (0,1) day/night range  

---

## 6. Features that matter (justified)

### 6.1 Primary (carry forward)

1. **`daily:range`** — July LOO #1–3; reconnects with classical размах; β likelihood.  
2. **`daily:mean + daily:range`** — stable top-10; \(\tau\) posterior ~6.5–6.8 on refits.  
3. **`daily:mean` alone** — still strong; slightly earlier/tighter \(\tau\) (~6.0) than range-only.  
4. **Profile grid:** prefer **`N=48`, overlap \(\in\{0,0.25\}\)`** for daily aggregates; keep **`N=24`, ov=0.5`** as the thesis-compatible bridge (`W=2,S=1`).

### 6.2 Secondary

5. **`day:range`** (half-profile) — frequent companion in top-50; treat as circadian-structure probe, not alone (half-day-only configs start ~rank 97).  
6. **Legacy `concat:mean`** under chunked \(\tau\) — still the 896-study champion; useful for **replication continuity** and preprint claims.  
7. **Night metrics** — present in top-100 but not dominant; useful for robustness, not headline.

### 6.3 Weak / poorly informative for shared-\(\tau\) search

8. **`shape_shift` / L1–L2 form distances** — negative ELPD scale; \(\tau\) unstable; descriptive only.  
9. **High-order moments** (skewness, kurtosis) — not needed for current winning models.  
10. **`std` as separate from range** on 2-point chunks — largely redundant with range.

### 6.4 Practical feature recipe

For the next confirmatory run:

```text
features: {daily: [range]} and {daily: [mean, range]}
N × overlap: (48,0.0), (48,0.25), (24,0.5)   # last = thesis bridge
likelihoods: mean ∈ {student_t, normal}; range ∈ {beta, interval_inflated_beta}
tau: marginalized, lower=3, upper=8
```

Optional continuity arm: `concat:mean` with \((N=24,\text{ov}=0.5)\) or deprecated `(W=2,S=1)`.

---

## 7. Poorly detectable profiles (justified)

### 7.1 High Pareto / repeatedly removed events (July full search)

Model indices on the **33 exported** profiles (`profile_cache/rem_n24_ov0.50_stage2/samples_10days_metadata.csv`):

| Model idx | Removal count (across configs) | Identity |
|-----------|--------------------------------|----------|
| **23** | **403** | **R2, 2024-09-30, after_reversed** |
| **6** | **298** | **R3, 2024-09-30, before** |
| 18 | 38 | R2, 2022-11-07, after_reversed |
| 24 | 36 (also high residual influence) | **R3, 2024-09-30, after_reversed** |
| 21 | 28 | R2, 2023-04-11, after_reversed |
| 10 | 26 | R1, 2025-07-02, before |

**Cluster diagnosis:** the **2024-09-30** event (R2 and R3, before *and* after) is the clearest systematic hard case. After-windows contribute disproportionately to Pareto-k — consistent with classical papers’ strong *post*-event sleep distortion (regime may not be a clean single pre-event \(\tau\)).

### 7.2 Incomplete / non-exported after windows

Never entered the 33-event likelihood (metadata `exported=False`):

- R2 2024-10-29 after_reversed (7 missing days)  
- R3 2025-01-23 after_reversed (2 missing)  
- R3 2025-03-14 after_reversed (8 missing)  

These are **data-availability failures**, not model failures — but they bias the after_reversed coverage.

### 7.3 Shape-shift hard cases

- **R2 2024-09-30** removed in **all 126** shape_shift configs (index 5 in that event list).  
- Secondary removals: R4/R3 2025-07-02 after windows, R3 2025-01-23, R2 2023-04-21.

### 7.4 Artifact-contaminated days (form study)

From `rem_profile_artifact_cleaned_study.tex`: row 0 day 1; rows 4–5 day 3; row 22 day 4. Until automated artifact rules exist, these days **pollute both features and reference profiles**.

### 7.5 Feature shapes that fail detection

- **Shape-only features** (shape_shift, L1/L2 to early reference) — weak formal evidence; Bayesian ELPD poor.  
- **Half-day-only models without daily aggregates** — low LOO rank.  
- **after_reversed** as if it were a mirrored “pre-event” changepoint — often Pareto-heavy; biologically a different process (recovery / post-stress).

---

## 8. Comparison with previous 896-config conclusions

### 8.1 Agreements (keep)

1. **Shared \(\tau\) near 6.5–6.7 days** survives the methodology rewrite and a different feature ontology.  
2. **Short, relatively fine profiles** beat coarse/heavy smoothing (old: \(W=2\); new: \(N=48\) or thesis-bridge \(N=24\)).  
3. **Range remains scientifically central** once the classical daily aggregate is restored — even though 896 ranked concat:mean first.  
4. **Pareto-k hygiene matters**; top models can be clean (\(k<0.7\)), but a few events dominate retries.  
5. **Not an operational quake forecast** — REM-associated state change only (unchanged caveat).

### 8.2 Contradictions / shifts (document explicitly)

| Topic | 896 / thesis | July recheck | Reading |
|-------|--------------|--------------|---------|
| Winning feature | `concat:mean` | `daily:range` | Different feature constructors; both OK for \(\tau\), **not interchangeable for LOO claims** |
| Preferred likelihood (mean) | student_t **and** lognormal | normal / student_t; lognormal demoted | MLE + July grid agree lognormal is a poor primary for normalized mean |
| Preferred resolution | \(W=2,S=1\) \(\to\) (24, 0.5) | (48, 0.0) for daily | Finer sampling helps daily scalars; (24, 0.5) remains the continuity bridge |
| Timing narrative | \(\tau\approx 6.6\) d | same | Classical 2–4 d papers measure a **later/larger** effect, not the same onset parameter |
| Shape features | not in 896 grid | tested; **lose** | Do not replace mean/range |

### 8.3 Implication for preprint / thesis wording

- Keep \(\tau\approx 6.6\) as the robust claim.  
- Soften any implication that **`concat:mean` is uniquely preferred**; state that under daily aggregates **range (β)** leads LOO, consistent with earlier neuroseismo manuscripts.  
- Note that July used a **rewritten observation pipeline**; absolute ELPD values are not comparable 1:1 to June CSV fingerprints without a frozen builder.

---

## 9. Open questions / next experiments

1. **Frozen confirmatory grid** (recipe in §6.4) with longer MCMC (as in top10 refits) and pre-registered event exclusions for 2024-09-30 sensitivity.  
2. Implement **zero-inflated beta** + **skew-normal**; re-rank a small grid.  
3. Map half-profile indices to **true light/dark** phases before interpreting day/night biology.  
4. Separate models for **before** vs **after_reversed** (or hierarchical animal/event effects) — after windows look like a different generative story.  
5. Finish **risk-model CV** (`run_bayesian_risk_cv.py`) with non-fallback quiet controls.  
6. Prospective holdout on new events using fixed `daily:range`, \(N=48\), ov=0, beta + \(\tau\) prior as above.  
7. Reconcile classical 2–4 d effect-size curves with Bayesian \(\tau\) via a **growth / two-changepoint** or gradual-transition model.

---

## 10. Artifact index

| Path | Role |
|------|------|
| `final_results/final_no_pareto.csv` | 896-config W/S results (thesis) |
| `seismic_pipeline_standalone/run_output_8day_parallel_full/exhaustive_search_parallel.csv` | July main LOO table (1224) |
| `…/top10_refits/` | Longer MCMC refits + posterior summaries |
| `…/likelihood_recommendation.json` | Likelihoods used in July full search |
| `run_output_8day_parallel_shape_shift/exhaustive_search_parallel.csv` | Shape-shift ablation (126) |
| `seismic_pipeline_standalone/run_output_likelihood_fit_diagnostics/` | MLE family diagnostics |
| `reports/rem_profile_artifact_cleaned_study.tex` | Form / L1–L2 study (13 Jul 2026) |
| `reports/seismic_risk_predictive_study.tex` | 24 h risk protocol (CV pending) |
| `run_output_old_vs_current_exhaustive/` | Old vs new code benchmark |
| `literature/old_neuroseismoreports/` | Classical REM–quake manuscripts |
| `seismic_pipeline_standalone/seismic_pipeline/docs/changepoint_exhaustive_config_report.md` | Config/prior/likelihood map |
| `seismic_pipeline_standalone/seismic_pipeline/bayesian/priors.py` | Prior factory |
| `seismic_pipeline_standalone/seismic_pipeline/config/changepoint_defaults.py` | REM grid + W/S deprecation |
| `RESEARCH_STATUS_2026-08-06.md` | High-level project status |
| `literature/latex/mmcs-sfedu_thesis7/preprint_seismic_stress.tex` | Preprint claims |

---

## 11. One-paragraph verdict

The July rewrite correctly abandons entangled \((W,S)\) windows for **`n_points` + overlap`**, and under that protocol the data favour **daily REM range (beta)** at **fine resolution (\(N=48\))**, while still recovering the same **~\(6.6\)-day** changepoint as the 896-config **`concat:mean`** study. Classical neuroseismo reports already pointed at **range** and shorter **2–4 day** contrasts; the Bayesian model estimates an **earlier regime onset**. Research priors should stay concentrated on **Normal/HalfNormal + student_t/normal for mean**, **Gamma-parameterized beta (and careful IIB) for range**, and **uniform discrete \(\tau\)**; **shape_shift** and **2024-09-30 / incomplete after** profiles should be treated as known failure modes rather than silent contaminants.
