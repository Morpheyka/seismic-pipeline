# Task packet — IIB / ν-split into Results+Discussion (2026-08-12)

**Scope:** targeted insert from existing confirmatory artifacts (no new MCMC).

**Delivered**
- Tables: `tables/iib_range_regime_summary.md`, `tables/student_t_nu_split_vs_shared.md`
- Figures: `fig:diag-iib` (IIB 02+06), `fig:diag-mean-nu` (ν-split 02+06) → `latex/images/`
- Prose: EN+RU Results/Discussion in chapters + `body_en.tex` / `body_ru.tex`
- Build: `latex/build.sh` → paper_en.pdf (8p), paper_ru.pdf (9p)

**Numbers locked to artifacts**
- IIB: unit-interval \(y=\mathrm{range}/2\), \(\tau_{\mathrm{MAP}}=7\)
- ν: shared `4f6e4c855d72864d` vs existing `4f6e4c855d72864d_nu_split` only
- Bands on figures: **90%**

**Follow-ups (optional)**
- New ν MCMC if desired; 50% band plots; full ν-grid
