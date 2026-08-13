# Progress

## 2026-08-11

- Выбран воркшоп ICDM 2026: **OWAD**
- Зафиксированы title, позиционирование (хайп только в мотивации), методы, IEEE LaTeX, структура 8 секций
- Создана ветка `workshop/icdm2026-owad` от `merge/conference-article-release`
- Создано дерево `literature/icdm2026_owad/` (plan, chapters, latex, …)
- Статус: brainstorming завершён

## 2026-08-11 — Introduction

- **Stage:** S1 Evidence → S4 Drafting (OWAD Intro)
- **Статус:** черновик готов, ожидает подтверждения пользователя
- **Файл:** `chapters/01_introduction.md`
- **RU-перевод:** `chapters/01_introduction.ru.md` (для чтения автором)
- **Объём:** ~4280 chars / 563 words (min_chars=2800)
- **Пользователь подтвердил:** да (2026-08-11)
- **Артефакты:** evidence-map, blueprint, evidence-coverage, task packet, chapter-architecture
- **Правило:** далее к каждой EN-главе сразу добавлять `*.ru.md`

### Capability-use audit

- Required skills: using-research-writing, paper-orchestration, evidence-driven-writing, writing-chapters, writing-core
- Skills actually used: same (literature-review not run as separate pass; pool reused from RU article)
- Inputs consumed: RU `01_introduction.md`, RU evidence map/blueprint, thesis `references.bib`, OWAD plan/outline
- Inputs not used and why: full literature-review search (existing verified pool sufficient); style_check.sh path not found in skills tree — used local pattern/citation checks
- Artifacts produced: Intro prose + plan/review/evidence artifacts
- Verification run: char/word count; banned-phrase scan; citation keys ⊆ bib (0 missing)
- Remaining risk: classical neuroseismo 2–4 day window still without formal bib key; Sinkovec2022 is indirect small-n support; Related Work still empty so some method-family depth deferred

## 2026-08-11 — Positioning rebalance (anomaly-first)

- Автор: слишком много места под predictive framing; центр — модель поиска аномалии, связанной с сейсмическим событием
- Предиктивное — только внешний контекст линии работ, не сюжет рукописи
- Обновлены Intro / Related Work / Data (EN+RU) и plan

## 2026-08-11 — Positioning update (preparatory → predictive)

- Уточнение автора: исследование подтверждает статистическую аномалию как подготовку к будущей предиктивной модели
- Обновлены Intro/Related Work (EN+RU), `project-overview.md`, `notes.md`
- Non-claim сохранён: оперативный предиктор в этой работе не строится

## 2026-08-11 — Related Work

- **Stage:** S1 Evidence → S4 Drafting
- **Статус:** черновик готов, ожидает подтверждения пользователя
- **Файлы:** `chapters/02_related_work.md`, `chapters/02_related_work.ru.md`
- **Объём EN:** ~2759 chars / 342 words (min_chars=1800)
- **Пользователь подтвердил:** да (2026-08-11; после правки preparatory→predictive)

### Capability-use audit (Related Work)

- Required skills: evidence-driven-writing, literature-review, writing-chapters, writing-core, paper-orchestration (task packet)
- Skills actually used: same
- Inputs consumed: locked 3-theme outline; Intro; evidence map; CrossRef/DOI checks for S20–S21
- Inputs not used and why: full scholar_search.py (script path not present); broader anomaly-detection surveys beyond CPD not added to avoid citation sprawl
- Artifacts produced: EN+RU Related Work, blueprint, task packet, evidence-coverage-related-work, bib append
- Verification run: char count; banned-phrase scan; citation keys ⊆ bib
- Remaining risk: classical site reports still paraphrased; OWAD “open-world anomaly” angle rests on CPD/regime-shift framing rather than a dedicated OWAD survey cite

## 2026-08-11 — Data

- **Stage:** S4 Drafting
- **Статус:** черновик готов, ожидает подтверждения пользователя
- **Файлы:** `chapters/03_data.md`, `chapters/03_data.ru.md`
- **Объём EN:** ~2820 chars / 409 words (min_chars=1800)
- **Пользователь подтвердил:** да (2026-08-11; после правки 34 окна и anomaly-first rebalance)

### Capability-use audit (Data)

- Required skills: writing-chapters, writing-core, paper-orchestration (task packet)
- Skills actually used: same (evidence-driven light; facts from RU chapter)
- Inputs consumed: RU data chapter, data blueprint, preparatory→predictive notes
- Inputs not used and why: experiment-results-planning (no results claimed here)
- Artifacts produced: EN+RU Data, task packet, blueprint
- Verification run: char/word count; citation keys ⊆ bib
- Remaining risk: hardware model strings copied from RU draft; July multi-animal dates stated as in source

## 2026-08-11 — Method

- **Stage:** S2/S4 Method drafting
- **Статус:** черновик готов, ожидает подтверждения
- **Файлы:** `chapters/04_method.md`, `chapters/04_method.ru.md`
- **Объём EN:** ~4067 chars / 554 words (min_chars=2800)
- **Добавлено по запросу автора:** явное обоснование выбора семейств для mean vs range; IIB у 1 — из-за range≈1 при узких бинах + min–max по 8-дневному окну
- **Пользователь подтвердил:** да (2026-08-11; после правки rationale IIB≈1)

- Required skills: writing-chapters (input→output), writing-core
- Skills actually used: same
- Inputs consumed: RU `03_methods.md`, experiment-protocol, latex methods snippets
- Artifacts: EN+RU Method, task packet, blueprint
- Verification: char count; citation keys ⊆ bib
- Remaining risk: numerical screening outcomes belong in Results; IIB-as-analysis-set stated as post-screen rule without dumping figures here

## 2026-08-11 — Experimental Setup

- **Статус:** черновик готов, ожидает подтверждения
- **Файлы:** `chapters/05_experimental_setup.md`, `chapters/05_experimental_setup.ru.md`
- **Объём EN:** ~2939 chars / 398 words (min_chars=1800)
- **Пользователь подтвердил:** да (2026-08-11; «Дальше»)

### Capability-use audit (Experimental Setup)

- Required skills: writing-chapters, writing-core
- Inputs: RU experiment, protocol rev.3
- Artifacts: EN+RU setup, task packet, blueprint
- Remaining risk: confirmatory details abbreviated for 8+2; full primary table deferred to Results

## 2026-08-11 — Results

- **Статус:** черновик готов, ожидает подтверждения
- **Файлы:** `chapters/06_results.md`, `chapters/06_results.ru.md`
- **Объём EN:** ~3971 chars / 541 words (min_chars=2200)
- **Пользователь подтвердил:** да (2026-08-11; после уточнения mean+range = два признака)

### Capability-use audit (Results)

- Required skills: experiment-results-planning, writing-chapters, writing-core
- Artifacts: protocol, traceability, table-schema, figure manifest, EN+RU Results
- Verification: char count; numbers copied from RU results (not invented)
- Remaining risk: figures not yet copied into `icdm2026_owad/figures/`; IEEE selection of 2–3 plots pending

## 2026-08-11 — Figures

- Добавлены plate (`tau_model.tikz` / `_en`) и два mean-графика (`fig_mean_only_by_lik`, `fig_mean_by_profile`)
- PDF пересобраны: EN ~6 стр., RU ~7 стр.

## 2026-08-11 — Abstract + PDF build

- Abstract EN/RU: `chapters/00_abstract.md`, `00_abstract.ru.md`
- Авторы (все — НИТЦ нейротехнологий ЮФУ): Пономарёв М.Б., Саевский А.И., Шепелев И.Е., Косенко П.О., Чебров Д.В., Кирой В.Н.
- PDF: `latex/paper_en.pdf`, `latex/paper_ru.pdf` (по ~6 стр., xelatex+natbib)
- Каркас: article twocolumn (IEEE cls пока без официального шаблона ICDMW)

## 2026-08-11 — Paradigm rewrite (mean-first + interpretation)

- Автор: убрать акцент на неинформативных range-семействах; центр — красивые распределения mean; интерпретация: аномалия по мозговому сигналу + признаки/гиперпараметры
- Переписаны Method, Setup, Results, Discussion, Conclusion; подправлены Intro/Related Work/Data; notes + figure priority

## 2026-08-11 — τ calendar alignment

- Напоминание автора: \(t=8\) = 1 день от события, \(t=1\) = 8 дней; календарная удалённость \(9-t\)
- Правки: Method (определение индекса); Results/Discussion IIB — перевод \(\mathbb{E}[\tau]\) в сутки от события

## 2026-08-11 — Discussion

- **Статус:** черновик готов, ожидает подтверждения
- **Файлы:** `chapters/07_discussion.md`, `chapters/07_discussion.ru.md`
- **Объём EN:** ~3875 chars / 552 words (min_chars=2200)
- **Пользователь подтвердил:** да (2026-08-11; после правки календарного чтения \(\tau\))

- Required skills: writing-chapters, writing-core
- Inputs: RU discussion, OWAD Results
- Remaining risk: July-band comparison from RU discussion abbreviated (different prior/run) to avoid mixing protocols

## 2026-08-11 — Conclusion

- **Статус:** черновик готов, ожидает подтверждения
- **Файлы:** `chapters/08_conclusion.md`, `chapters/08_conclusion.ru.md`
- **Объём EN:** ~1028 chars / 142 words (min_chars=800)
- **Пользователь подтвердил:** нет

### Capability-use audit (Conclusion)

- Required skills: writing-chapters, writing-core
- Inputs: RU conclusions + OWAD Discussion/Results framing
- Remaining risk: Abstract still unwritten; IEEE LaTeX assembly pending

## 2026-08-11 — figures rescale + gallery
- Plate уменьшен (tikz compact + 0.78 textwidth).
- Histograms: крупнее шрифт, figure*.
- Gallery fig3: figures/gallery_fig3/A–O (15 вариантов + stacked hist).
- В PDF временно стоит A_lines_improved как fig:mean-profile.

## 2026-08-12 — best mean-only refit diagnostics
- Fingerprint `4f6e4c855d72864d` (daily:mean, student_t, N=24, ov=0), τ∈{2…8}, tune=6000/draws=3000/chains=4.
- Artefacts: `seismic_pipeline_standalone/run_output_8day_density_safe/refit_best_mean_only/rank11_4f6e4c855d72864d/` (`trace.zarr`, observations, summary).
- Figures: `literature/icdm2026_owad/figures/refit_best_mean_only/01_traces` … `05_likelihood_fan`.
- MCMC: R̂≈1.00; E[τ]≈5.74; MAP τ=7.

## 2026-08-12 — skew-normal + IIB range refit diagnostics
- Skew-normal mean-only: fingerprint `0da9bbc60553996e` (N=24, ov=0.5); E[τ]≈6.39, MAP τ=7.
  Figures: `figures/refit_skew_normal_mean/`.
- IIB range-only: fingerprint `00a650a4c85c7621` (N=12, ov=0); E[τ]≈6.40, MAP τ=7.
  Figures: `figures/refit_iib_range/`.
- Script `plot_refit_diagnostics.py` обобщён на student_t / skew_normal / IIB.
- В `diagnostics.py` добавлен PDF для `skew_normal`.

## 2026-08-12 — skew E[τ] modes + IIB obs scale fix
- Skew mean-only E[τ] bimodal: low ~5.0–5.6 (day/night) vs high ~6.1–6.5 (daily).
  - Mode1: `036cc0ffcaf8c608` night:mean → `figures/refit_skew_mode1_low/`
  - Mode2: `65515c6970fa7603` daily:mean → `figures/refit_skew_mode2_high/`
- IIB range plots: observations scaled by `support_upper=2` to unit interval (match likelihood).

## 2026-08-12 — IIB/ν-split diagnostics into Results+Discussion

- **Сделано:** сводка IIB (`tables/iib_range_regime_summary.md`) + shared vs existing ν-split (`tables/student_t_nu_split_vs_shared.md`); фигуры `fig:diag-iib` / `fig:diag-mean-nu` в `latex/images/`; текст EN+RU в `chapters/06|07` и `body_*.tex`; PDF пересобраны.
- **IIB (\(y=\mathrm{range}/2\), \(\tau_{\mathrm{MAP}}=7\)):** before median \(0.736\), \(\ge 0.9\): \(20.7\%\); after median \(0.699\), \(\ge 0.9\): \(12.1\%\); event-wise drop \(20/33\).
- **ν-split (без нового MCMC):** shared \(\mathbb{E}[\tau]\approx 5.74\), \(\mathbb{E}[\nu]\approx 20.7\); split \(\mathbb{E}[\tau]\approx 6.01\), \(\nu_1\approx 29.8\), \(\nu_2\approx 13\); σ-gap \(0.017\to 0.006\).
- **PDF:** `latex/paper_en.pdf` (8 стр.), `latex/paper_ru.pdf` (9 стр.); build OK.
- **Follow-ups:** опциональный новый ν-прогон; полоса \(50\%\) на диагностиках; полный \(\nu\)-grid — не блокеры текущего текста.

## 2026-08-12 — Conference-style review `paper_en.pdf`

- **Артефакт:** `plan/review_paper_en.md`
- **Вердикт:** major revision / риск desk reject (нет anonymization + не ICDM IEEE template)
- **Топ-дыры:** float’ы Figs 3–5; нет LOO-таблицы; failed shuffle vs сильный detection claim; кириллица Rodkin/GOST в EN cite; слабый OWAD fit; `L-ACRD`/`Passyuk`/`Sanford2003` hygiene
- Skills: peer-review + writing-core + verification (pdftotext, cite∩bib, log, OWAD CFP)

## 2026-08-12 — Revision lock: within-window null + claim

- Brainstorm decisions written into `plan/notes.md`, `experiment-protocol.md` (this entry).
- **Mandatory null:** within-window day-shuffle (permute valid days inside each window; keep NaN masks).
- **Delete from paper plot:** old proxy/date-shuffle.
- **Viz:** Variant A only (`fig:null-within-window`).
- **Primary arm:** `4f6e4c855d72864d` daily:mean student_t \(N=24\) ov=0; seeds `0,1,2`.
- **Claim:** event-aligned day order (not calendar-date separation / not forecast).
- Science exec: script + run → `within_window_shuffle/` + fig A (IEEE/LOO/prose = other agents).

## 2026-08-12 — Within-window day-shuffle run (science)

- **Script:** `seismic_pipeline_standalone/scripts/run_within_window_day_shuffle.py`
- **Plot:** `seismic_pipeline_standalone/scripts/plot_within_window_shuffle_variant_a.py`
- **Results:** `seismic_pipeline_standalone/run_output_8day_density_safe/within_window_shuffle/`
- **Figure:** `literature/icdm2026_owad/latex/images/fig_null_within_window.{svg,pdf,png}` (+ `figures/null_within_window/`)
- **MCMC:** confirmatory tune=6000 / draws=3000 / chains=4 for shuffle seeds; real PMF from stored `trace.zarr` (same confirmatory budget).
- **Real mean \(p(\tau=k)\):** peak mid-window (τ=6–7); E[τ]≈5.74; MAD→uniform≈0.086
- **Shuffle seeds mean:** **not uniform** — residual peak at τ=7 (≈0.42); MAD→uniform≈0.082 (barely flatter than real). Seed-wise: s0 flatter; s1/s2 lock onto τ=7.
- **Blocker for claim:** success criterion “shuffle ≈ uniform” **failed** under this arm/seeds; prose should weaken event-order claim or note residual structure under within-window permutation.

## 2026-08-12 — prose-claim rewrite (post failed null)

- Removed all proxy date-shuffle / “shuffle does not separate dates” language from paper prose.
- Inserted within-window control + Fig `fig:null-within-window` with residual τ≈7 mass / failed uniformity.
- Weakened claim: mid-window onset **not uniquely** explained by event-aligned day order.
- LOO table + IEEE wrapper preserved. `paper_en.pdf` = 8 pages after rebuild.
- `prose-claim` done; `rebuild-verify` left for later if floats still moving.

## 2026-08-12 — rebuild-verify

- **Build:** `PAPER_CAMERA=1 ./build.sh` (xelatex+bibtex) → OK; `paper_en.pdf`, `paper_en_camera.pdf`, `paper_ru.pdf`.
- **Page count:** `paper_en.pdf` = **8** (letter); camera = 8. Limit 8+2 → pass.
- **Checklist (§0 / plan waves):**
  - Anonymous submission path (`paper_en.tex` / IEEEtran, Anonymous Authors, no date) → **pass**
  - Named camera copy (`paper_en_camera.tex` + PDF with author blocks) → **pass**
  - No proxy/date-shuffle language in `body_en.tex` → **pass**
  - Null fig present + referenced (`fig:null-within-window` = Fig. 5, p.6) → **pass**
  - LOO Table I top-10 present (p.4) → **pass**
  - Floats not inside References (Figs 1–5 on pp.4–6; Fig. 6 on p.7 with Discussion; no captions after References heading) → **pass** (not catastrophic)
  - IEEE conference class + `IEEEkeywords` → **pass** vs old article+geometry
- **No build blockers fixed** (no undefined refs / missing images / build errors).
- **Remaining risks:** Fig. 6 shares p.7 with Discussion/Conclusion/References start (tight but OK); OWAD Related Work depth / other §1 science items not in this verify scope; null uniformity failed (claim already weakened in prose).
- **Todo:** `rebuild-verify` completed.

## 2026-08-12 — Fig. 2 \(\mathbb{E}[\tau]\) glyph

- Re-exported `latex/images/fig_mean_only_by_lik.{pdf,png}` from `figures/data/mean_only_long.csv` with STIX mathtext + `pdf.fonttype=42`; rebuilt `paper_en.pdf` — Fig. 2 axis \(\mathbb{E}[\tau]\) readable (was broken Type3/dejavusans).

## 2026-08-13 — Null-control suite (exps 4→3→1→2)

- Script: `seismic_pipeline_standalone/scripts/run_null_control_suite.py`
- Outputs: `run_output_8day_density_safe/null_control_suite/` + figs `figures/null_control_suite/`
- **Exp4 synthetic:** aligned recovers τ=6 (MAD≈0.245); within-window shuffle flattens (MAD≈0.04–0.07) → **null has power**
- **Exp3 i.i.d.:** peaks scatter (7/2/4); E[τ] leaves mid-window band
- **Exp1 20 seeds ww-shuffle:** peak_counts {2:8, 3:3, 7:3, 8:6}; frac peak τ=7 = 0.15 — earlier 3-seed mean-PMF was misleading
- **Exp2 column-shuffle:** preserves column multisets → day means unchanged → E[τ]≈5.74–5.78 peak τ=7 like real (weak null for shared-τ)

## 2026-08-13 — prose rewrite (post suite)

- **Claim restored/corrected:** mid-window onset **sensitive to within-window day order** (20-seed peak tallies + Exp4 power); calendar causality still weak.
- **Do not say** “null failed to flatten / residual τ=7 ≈0.42” — that was mean-PMF over 3 seeds.
- Updated: `body_en.tex`, `body_ru.tex`, abstracts (`paper_en`, camera, `paper_ru`, chapter mirrors), Discussion/Conclusion EN+RU chapters, Setup, Results null subsection.
- Fig `fig_null_within_window`: left real PMF, right peak-τ histogram over 20 seeds.
- Rebuilt `paper_en.pdf` / `paper_ru.pdf`; camera via `PAPER_CAMERA=1`.

## 2026-08-13 — OWAD fit + title (items 1–2)

- **Stage:** S1 Evidence → S4 Drafting (targeted; single-agent degraded mode OK for non-full redraft).
- **Title:** `Bayesian Changepoint Detection for Event-Aligned REM Means Near Seismic Events` (RU synced).
- **OWAD prose:** new RW theme + Intro bridge; cites Chandola2009, Gama2014, Han2023, Faber2024 (CrossRef-verified). Explicit non-claim: not continual / normality-shift adaptation.
- **Artifacts:** task packet `owad_fit_title.md`; evidence map S22–S25; blueprints Intro/RW; `paper_en.pdf` still **8** pages.
- Spec review: title matches mean-first; OWAD cites present; no fabricated refs.
- Quality review: venue fit stated as setting+case study, not overclaim of workshop methods.

### Capability-use audit
- Required skills: paper-orchestration, evidence-driven-writing, literature-review, verification
- Skills actually used: those four (writing-core compression implicit)
- Inputs consumed: review_paper_en §0/§1 title+OWAD; OWAD CFP; CrossRef DOIs; body_en/ru
- Inputs not used: full multi-agent chapter dispatch (targeted revision, not full redraft)
- Artifacts produced: bib keys, RW paragraph, title, chapter mirrors, evidence coverage
- Remaining risk: classical site-report bib still missing; page budget tight if further RW growth

## 2026-08-13 — Author facts (item 3) + IIB n=34 recompute

- **Facts locked:** *Rattus norvegicus*; ADC `L-CARD`; ethics protocol No. 1 (1 Apr 2022); center EN name; no classical 2–4 day cite yet.
- **IIB 20/33 root cause:** `run_top10_refit_plots.py` used `drop_incomplete_events=True` → dropped `R3 / 2025-01-23 / after_reversed` (2 missing days) which **survives** day-mask \(K=6\). Screening/paper cohort is \(n=34\).
- **Fix:** `--density-safe` path in `run_top10_refit_plots.py`; refit `refit_iib_range_n34/` (chains=4). Event-wise **20/34**; \(\mathbb{E}[\tau]\approx 6.88\), MAP \(\tau=7\); regime medians/shares updated; Fig. diag-iib regenerated.
- July R1–R4 windows already present in \(n=34\) (no missing animal).

## 2026-08-13 — Melochi (item 4) + commit prep

- Paired \(\Delta\mathbb{E}[\tau]\): mean paired \(+0.46\) (was ambiguous \(+0.42\)); pooled \(n=84\) wording.
- Data: Methods→Method; Data/Code Availability section (anon-safe).
- Bib: Rodkin EN primary; GOST English agency form; Adams `@misc` arXiv; `Sanford2010`→`Sanford2003`.
- `paper_en.pdf` still 8 pages.
