# Research / Change Diary

Append-only log of substantial changes and experiments: what, why, result.

## 2026-08-07 — Publishable full-rerun (density-safe rev.3): код + smoke + фоновые прогоны

- **Change:** Заморожен `experiment-protocol.md`; модуль `features/day_mask.py` (corrected ARTIFACT map `{1,2}` для R2 2023-04-21 after_reversed; K=6); NaN day-slots в `rem_profile_calculator`; NaN-skip в `changepoint_model`; `skew_normal` + настоящий `zero_inflated_beta` (пресет `range_zoib` исправлен); скрипты `run_parallel_search_8day_density_safe.py`, `run_density_safe_confirmatory.py`, `run_density_safe_neg_control.py`; главы 2–7 и LaTeX переписаны под новый протокол.
- **Why:** July plain Beta / N=48 / τ≈6.5–6.7 не publishable; нужен density-safe rerun с единым day-mask.
- **Result:** Smoke OK: export 36→cohort **n=34**, masked_days=8. Confirmatory **завершён** (12 ячеек): Primary A mask ON E[τ]=**4.25**; mask OFF **3.16**; before_only **2.11**; B BC/ZOIB разброс 2.0–5.9; DIAG plain beta elpd/fe=4.55 > BC. Neg-control seeds 0–2 без резкого null-контраста. Full exploratory **ещё бежит** (`run_output_8day_density_safe/`, resume в `RESUME.md`). PDF: `literature/conference_article_ru/latex/conference_article.pdf` (4 стр., xelatex).
- **Artifacts:** `run_output_density_safe_confirmatory/confirmatory_results.csv`, `run_output_density_safe_neg_control/`, `reports/figures/density_safe/`, `run_output_8day_density_safe/RESUME.md`
- **Next:** Дождаться full CSV; при необходимости удлинить MCMC / календарный shuffle; обновить T3 visit rates.

## 2026-08-07 — Сборка статьи + перепроверка литературы

## 2026-08-07 — Сборка статьи + перепроверка литературы

- **Change:** Собран NeurIPS-RU PDF из глав; CrossRef-проверка всех cite DOI; ослаблены ссылки на Vyazovskiy2005 (REM-тета → Fang/Saevskiy); добавлен DOI Rodkin2011.
- **Why:** Пользователь: «Собирай. А ты перепроверил литературу?»
- **Result:** `literature/conference_article_ru/latex/conference_article.pdf` (6 стр.); `refs/lit-verification.md`. DOI цитируемых статей OK; Sanford2010=2003; Vyazovskiy тематически слаб для REM-скоринга.
- **Artifacts:** `latex/conference_article.tex`, `.pdf`, `refs/lit-verification.md`
- **Next:** Правка семейств правдоподобий → rerun → перепись Results; опционально rename Sanford2010→2003.

## 2026-08-07 — Глава «Выводы»

- **Change:** `chapters/07_conclusions.md` — 5 пунктов: модель+LOO; устойчивость τ; range/mean+range + caveat Beta; τ≠окно 2–4 сут; консервативная интерпретация + rerun после likelihood.
- **Why:** Закрытие структуры статьи.
- **Result:** Черновик согласован с Results/Discussion; без диссертации.
- **Artifacts:** `chapters/07_conclusions.md`
- **Next:** ОК пользователя; сборка NeurIPS tex из глав.

## 2026-08-07 — Глава «Обсуждение»

- **Change:** `chapters/06_discussion.md`: τ vs окно эффекта 2–4 сут; caveat Beta; влиятельные окна без auto-delete; primary как рамка; ограничения; финальный rerun после смены likelihood.
- **Why:** Следующий раздел после принятого черновика Results.
- **Result:** Черновик без диссертации и без раздувания инверсии τ.
- **Artifacts:** `chapters/06_discussion.md`
- **Next:** Выводы.

## 2026-08-07 — Глава «Результаты» (только N×ov)

- **Change:** `chapters/05_results.md`: устойчивость τ (Q1≈6.70, top-50≈6.55), top-таблица daily range/mean+range, caveat plain Beta у границы, shape secondary, 2024-09-30 → обсуждение. `tables/table-schema.md`.
- **Why:** Центр статьи — итоги полного перебора без диссертации.
- **Result:** Черновик с флагом переписи после смены likelihood.
- **Artifacts:** `chapters/05_results.md`, `tables/table-schema.md`
- **Next:** Обсуждение (primary, классика 2–4 сут, влиятельные окна).

## 2026-08-07 — Только сетка N×overlap; без упоминаний диссертации

- **Change:** Из глав 2–4 и plan убраны диссертация / 896 / W/S-перебор; единственный экспериментальный сюжет — полный перебор \(N\times\mathrm{ov}\) с daily mean/range.
- **Why:** Диссертация не опубликована; нельзя ссылаться и упоминать.
- **Result:** Методы и Эксперимент переписаны под одну методологически целевую сетку.
- **Artifacts:** `chapters/02_data.md`, `03_methods.md`, `04_experiment.md`, `plan/experiment-protocol.md`, `outline.md`, `project-overview.md`
- **Next:** Результаты только по July-прогону (после ОК).

## 2026-08-07 — Глава «Эксперимент (полный перебор)»

- **Change:** Черновик `chapters/04_experiment.md` + `plan/experiment-protocol.md`. Две сетки (896 и July); акцент на daily mean/range; legacy half-profile кратко; флаг rerun после смены likelihood.
- **Why:** Центр статьи по согласованному плану.
- **Result:** Дизайн без dump Results; запрет кросс-builder elpd; без LOO-IC.
- **Artifacts:** `chapters/04_experiment.md`, `plan/experiment-protocol.md`
- **Next:** Результаты после ОК.

## 2026-08-07 — Глава «Методы»

- **Change:** Черновик `chapters/03_methods.md`: профили W/S и N×ov, признаки, модель разладки, likelihoods, PSIS-LOO/MCMC. Сетки перебора отложены в гл. 4.
- **Why:** Пользователь подтвердил переход к методам.
- **Result:** Конференционный объём; без акцента на инверсию τ; календарная интерпретация чисел — в Results.
- **Artifacts:** `chapters/03_methods.md`, `plan/chapter-blueprints/03_methods-blueprint.md`
- **Next:** Гл. 4 «Эксперимент (полный перебор)» после ОК.

## 2026-08-07 — Глава «Данные»

- **Change:** Черновик `chapters/02_data.md` по диплому (площадка, гипнограмма, 14 дат, окна before/after_reversed, July n=33).
- **Why:** Следующий раздел после подтверждённого Введения.
- **Result:** Конференционный объём без dump всей таблицы событий; профили отложены в Методы.
- **Artifacts:** `chapters/02_data.md`, `plan/chapter-blueprints/02_data-blueprint.md`
- **Next:** Методы после ОК пользователя.

## 2026-08-07 — Правка Введения (согласованность, без акцента на инверсию τ)

- **Change:** Переписан `chapters/01_introduction.md`: выровнены формулировки, убран развёрнутый абзац про инверсию τ; правило оставлено только в `plan/`.
- **Why:** Замечания пользователя о несогласованности и лишнем упоре на индексацию.
- **Result:** Более ровная проза; estimand τ vs классика 2–4 сут сохранён без технической лекции об индексе.
- **Artifacts:** `chapters/01_introduction.md`; обновлены blueprint и project-overview.

## 2026-08-07 — Введение конференционной статьи (evidence-driven)

- **Change:** Evidence-map, blueprint и черновик Введения для `literature/conference_article_ru/`.
- **Why:** Старт нормальной статьи по скиллам; центр сюжета — полный перебор; инверсия τ явно в Intro.
- **Result:** `chapters/01_introduction.md` (6 абзацев IMRAD-вступления); артефакты в `refs/` и `plan/chapter-blueprints/`.
- **Artifacts:** `refs/evidence-map.md`, `plan/chapter-blueprints/01_introduction-blueprint.md`, `plan/review/evidence-coverage.md`, `chapters/01_introduction.md`
- **Next:** Правки пользователя → Данные или Методы.

## 2026-08-07 — План конференционной статьи (RU)

- **Change:** Завершён brainstorming; создан проект `literature/conference_article_ru/` (`plan/`, `chapters/`). Центр сюжета — полный перебор; primary A/B в обсуждении; зафиксирована инверсия индекса τ (8→1 сут до события).
- **Why:** Нормальная конференционная статья вместо коротких карточек Primary A/B.
- **Result:** Согласованы тип/название/структура; каркас NeurIPS RU; конкретный cls позже.
- **Artifacts:** `literature/conference_article_ru/plan/project-overview.md`, `outline.md`, `progress.md`, `chapters/01–07_*.md`
- **Next:** Evidence-map + написание с Введения (или главы по выбору пользователя).

## 2026-08-07 — Препринты Primary A и Primary B (NeurIPS, RU)

- **Change:** Два отдельных двухколоночных препринта под пререгистрированные конфиги: `preprint_primary_A.tex` (mean+range, N=24, ov=0.5) и `preprint_primary_B.tex` (range-only тройка + beta_constrained). Общий `preamble_common.tex`. Цифры из integrity confirmatory.
- **Why:** Развести заголовки A и B; не смешивать cherry-pick по ELPD; влиятельные события оставлены в primary.
- **Result:** PDF A — 4 стр.; PDF B — 3 стр.; 2 колонки (geometry+twocolumn). A: E[τ]=6.285; B: 7.254 / 7.097 / 5.747.
- **Artifacts:** `literature/latex/preprint_neurips/preprint_primary_{A,B}.{tex,pdf}`, `preamble_common.tex`
- **Next:** При желании расширить фигурами posterior τ / вставить в общий обзорный препринт.

## 2026-08-06 — Integrity confirmatory MCMC + sensitivity τ + negative control

- **Change:** Прогнаны Primary A/B confirmatory MCMC (`beta_constrained`, не plain beta), заполнена таблица чувствительности τ, страты toxic/before/drop, полный negative-control MCMC (`shuffle_dates` rebuild). Добавлен `scripts/run_integrity_confirmatory.py`; исправлен `run_negative_control_shuffle.py` (корректный export+fit, `support_upper=2.0`).
- **Why:** Закрыть unchecked пункты `reports/research_integrity_checklist.md` (§1–6) перед заявлениями о победителях признаков.
- **Result:** Primary A (`daily:mean+range`, N=24,ov=0.5, student_t+beta_constrained, full, n=33): E[τ]=**6.285**, HDI₆₀w=1.342, elpd/fe=**3.034**; Pareto>0.7: R3/R2 **2024-09-30**. Primary B тройка beta_constrained: E[τ]=7.254 / 7.097 / 5.747. Диапазон E[τ] по сетке (без plain beta): **[5.747, 7.687]**. Drop 2024-09-30: E[τ]=**4.525** (Δ≈−1.76). before_only: E[τ]=**3.880**. Neg-ctrl seed0: E[τ]=**5.265**, HDI₆₀w=**3.625** (τ PASS); elpd/fe=3.051 ≈ primary (ELPD inconclusive). Primary страта = **full**.
- **Artifacts:** `run_output_integrity_confirmatory/integrity_sensitivity_full_seed20260806.{csv,json}`; `integrity_primary_a_only_{drop_2024_09_30,before_only}_seed20260806.json`; `run_output_negative_control/mcmc_result_seed0_shuffle_dates.json`; `reports/research_integrity_checklist.md`; `scripts/run_integrity_confirmatory.py`
- **Next:** ≥3 seeds negative control; artifact-day mask sensitivity (§3.3); опционально усилить MCMC budget / GPU jaxlib.

## 2026-08-06 — NeurIPS preprint: 2 колонки + расширение методов

- **Change:** Исправлен layout `literature/latex/preprint_neurips/`: официальный `neurips_2025.sty` одноколоночный (`textwidth=5.5in`); добавлен override geometry (`7in`) + `\twocolumn[{title/abstract}]`; `\@notice` переведён с float на footnote (иначе `Not in outer par mode`). Существенно расширены разделы данные/профили/модель/поиск/результаты/integrity (таблицы 896/July/beta-smoke, рис. rem_profile, tau_model, q2_tau, hist_elpd, top100).
- **Why:** Жалоба на 1-колоночный PDF и слишком тонкий 4-стр. stub; нужны детали методов и оговорки без уникальных LOO-победителей.
- **Result:** PDF **7 стр.** letter, визуально **2 колонки**; без LaTeX Error. Claims: τ≈6.5–6.7 устойчив; rank зависит от сетки; plain-beta range ELPD — boundary artifact.
- **Artifacts:** `literature/latex/preprint_neurips/preprint_seismic_stress_neurips.tex`, `.pdf`; mmcs `preprint_seismic_stress.tex` не трогали.
- **Next:** Confirmatory MCMC primary A/B + заполнение sensitivity τ table.

## 2026-08-06 — Перевод research_integrity_checklist на русский

- **Change:** Полный перевод `reports/research_integrity_checklist.md` на русский (структура, таблицы, чеклисты сохранены; идентификаторы кода без перевода).
- **Why:** Правило пользовательских отчётов на русском.
- **Result:** Файл перезаписан; содержание то же.
- **Artifacts:** `reports/research_integrity_checklist.md`

## 2026-08-06 — Research integrity checklist + negative-control stub

- **Change:** Added `reports/research_integrity_checklist.md` (plan §5): pre-registered primary configs, τ sensitivity protocol table, toxic-profile hygiene, cross-builder ELPD ban, negative-control protocol. Implemented `scripts/run_negative_control_shuffle.py` (shuffle_dates / permute_labels; dry-run default; optional `--run-mcmc`).
- **Why:** Freeze analysis-candidate configs after beta-boundary audit showed plain-beta `daily:range` ELPD leadership is largely a boundary artifact; need τ sensitivity + null control before paper claims.
- **Result:** Checklist ready. Primary configs: (A) `daily:mean+range` N=24 ov=0.5 + student_t/normal + **beta_constrained**/IIB; (B) `daily:range` with **beta_constrained**/IIB (not plain beta). Dry-run smoke OK: shuffle_dates seed0 changed 28/36 dates (78%); permute before-only seed1 changed 15/18. Full MCMC null fit deferred (expensive / needs env).
- **Artifacts:** `reports/research_integrity_checklist.md`; `seismic_pipeline_standalone/scripts/run_negative_control_shuffle.py`; `seismic_pipeline_standalone/run_output_negative_control/events_*.json`; refs `reports/beta_boundary_elpd_audit.md`, `reports/july_recheck_methodology_analysis.md`.
- **Next:** Fill sensitivity τ table cells; run `--run-mcmc` negative control for Primary A.

## 2026-08-06 — NeurIPS Russian preprint (`preprint-corea`)

- **Change:** New 2-column NeurIPS 2025 preprint in Russian (`literature/latex/preprint_neurips/`), style from official NeurIPS2025 Styles.zip; left `preprint_seismic_stress.tex` (mmcs_article) untouched. Claims rewritten from July recheck + beta-boundary audit; terms aligned with old neuroseismo reports (размах профиля ПС, \(N{=}12,\mathrm{ov}{=}0\)).
- **Why:** Plan todo `preprint-corea` with venue override to NeurIPS (not IEEE); must not claim unique LOO winners; separate classical 2–4 d effect window from Bayesian onset \(\tau\sim 6.6\).
- **Result:** PDF builds with XeLaTeX + polyglossia/Liberation (4 pages letter). Robust claim: \(\mathbb{E}[\tau]\approx 6.5\)–\(6.7\); feature LOO rank grid/likelihood-dependent; plain-beta range ELPD treated as boundary artifact (prefer constrained/IIB/mean+range). Conservative: not stress assay, not EQ forecast.
- **Artifacts:** `literature/latex/preprint_neurips/preprint_seismic_stress_neurips.tex`, `.pdf`, `neurips_2025.sty`; compile via `xelatex` → `bibtex` → `xelatex`×2.
- **Next:** Integrity checklist / sensitivity τ table if still open in plan.

## 2026-08-06 — ELPD audit + beta boundary constrained prior

- **Change:** Audited ELPD/LOO columns (`elpd_loo`, `elpd_loo_per_feature_event`, `loo_ic=-2*elpd`); added unit tests and CSV legend; implemented `beta_constrained` (α,β≥1 via `gamma_offset`) keeping IIB; diagnostic script + smoke refits vs plain beta.
- **Why:** July winner `daily:range`+plain beta may be boundary-density ELPD inflation (β≪1, range/2 mass near 1), not physiology; need to separate metric correctness from artifact.
- **Result:** ELPD sign OK (not a bug). Boundary artifact **confirmed**: ~25–30% of daily range/2 in [0.9,1]; top-refit β₁≈0.52; corr(y,LL)≈0.81–0.88. Smoke N=48,ov=0: plain beta elpd/feat·evt **6.62** → constrained **5.18** / IIB **4.83**; E[τ] stays ~6.2–7.7. Verdict: range ELPD leadership is largely an artifact; τ stable.
- **Artifacts:** `reports/beta_boundary_elpd_audit.md`; `scripts/diagnose_beta_boundary.py`; `tests/test_elpd_sign.py`; `tests/test_beta_constrained.py`; `run_output_8day_parallel_full/beta_boundary_diag/beta_boundary_diagnostics.json`; priors/changepoint_model/search_export updates.
- **Next:** Integrity checklist + sensitivity τ table across mean/range × constrained/IIB; preprint wording must not claim unique `daily:range`+beta winner.

## 2026-08-06 — Preprocess/feature pytest smoke (`preprocess-smoke`)

- **Change:** Added pytest suite under `seismic_pipeline_standalone/tests/` for REM grid geometry, (W,S)↔(N,ov) equivalence, daily mean/range + `support_upper=2`, even/odd and day/night splits; wired shape_shift + IIB smoke scripts. Restored missing `seismo/rem_shape_shift.py`; fixed `scripts/test_shape_shift_feature.py` to pass `n_points_per_day`.
- **Why:** Plan todo `preprocess-smoke` — empty tests/ blocked confidence in July n_points/overlap + daily aggregates.
- **Result:** 10/10 targeted smokes PASS (`pytest tests/test_rem_preprocess_smoke.py tests/test_shape_shift_smoke.py tests/test_interval_inflated_beta_smoke.py`). Bugs found: missing `rem_shape_shift` module (script ImportError); shape_shift smoke omitted required `n_points_per_day`.
- **Artifacts:** `tests/conftest.py`, `tests/test_rem_preprocess_smoke.py`, `tests/test_shape_shift_smoke.py`, `tests/test_interval_inflated_beta_smoke.py`, `seismo/rem_shape_shift.py`
- **Next:** Integrity checklist / beta-boundary rerank (other plan todos).

## 2026-08-06 — Russian translation of July recheck report

- **Change:** Full Russian rewrite of `reports/july_recheck_methodology_analysis.md` (overwrite original English version); preserved markdown structure, tables, numbers, paths, and English code identifiers.
- **Why:** User requested Russian scientific version of the July n_points/overlap methodology synthesis for local use and thesis/preprint alignment.
- **Result:** Report fully translated; technical conclusions unchanged (\(\tau\approx 6.5\)–\(6.7\) d; `daily:range` at N=48; `shape_shift` secondary).
- **Artifacts:** `reports/july_recheck_methodology_analysis.md`

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
