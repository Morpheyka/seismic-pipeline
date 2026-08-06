# Research / Change Diary

Append-only log of substantial changes and experiments: what, why, result.

## 2026-08-06 — July n_points/overlap recheck synthesis

- **Change:** Detailed analysis of July rewritten methodology (`n_points`+`overlap` vs fixed W/S), July exhaustive/shape_shift runs, old neuroseismo manuscripts, and comparison to 896-config thesis results. Wrote `reports/july_recheck_methodology_analysis.md`; allowed diary/status tracking in `.gitignore`.
- **Why:** User recalled abandoning fixed windows for point-count/overlap during the proper-methodology recheck; needed justified priors/features/poor-profile conclusions before resuming work.
- **Result:** \(\tau\approx 6.5\)–\(6.7\) d **confirmed**. Feature winner **shifts**: thesis `concat:mean` → July **`daily:range` (beta)**, best at **N=48, ov=0**. `shape_shift` weak (ELPD/feat \(\lt 0\), \(\tau\sim 5\)). Hard cases: **2024-09-30 R2/R3** (Pareto), incomplete after_reversed windows, artifact days from rem-profile study. Classical manuscripts already privileged **размах** and 2–4 d contrasts (different estimand than Bayesian onset).
- **Artifacts:** `reports/july_recheck_methodology_analysis.md`; `seismic_pipeline_standalone/run_output_8day_parallel_full/exhaustive_search_parallel.csv`; `run_output_8day_parallel_shape_shift/exhaustive_search_parallel.csv`; `literature/old_neuroseismoreports/`; `docs/changepoint_exhaustive_config_report.md`
- **Next:** Confirmatory frozen grid `{daily:range}` + `{daily:mean,range}` at (48,0)/(48,0.25)/(24,0.5); implement zero-inflated beta + skew-normal; sensitivity drop of 2024-09-30; finish risk CV.

## 2026-08-06 — Status recovery after ~2 week pause

- **Change:** Reviewed repo state; wrote `RESEARCH_STATUS_2026-08-06.md`. Created personal Cursor skill `research-change-diary` and user rule for diary logging across projects.
- **Why:** Restore context after break; prevent loss of experiment memory going forward.
- **Result:** Core Bayesian 896-config REM changepoint study is complete and written up (thesis + preprint, \(\mathbb{E}[\tau]\approx 6.64\) days). Next frontier is 24h risk model: dataset exists (`run_output_risk_model/`), CV metrics not yet produced; quiet controls use fallback (`n_quiet_pool_strict=0`).
- **Artifacts:** `RESEARCH_STATUS_2026-08-06.md`; preprint `literature/latex/mmcs-sfedu_thesis7/preprint_seismic_stress.tex` / `preprint_ex.pdf`; results `final_results/final_no_pareto.csv`; risk protocol `reports/seismic_risk_predictive_study.tex`
- **Next:** Run `run_bayesian_risk_cv.py`; fix/strictify quiet-control sampling; then geophysical covariates.
