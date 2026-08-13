# Evidence map — Introduction (OWAD)

Pooled from `conference_article_ru` evidence map + thesis bib. Facts limited to title/abstract/DOI notes or prior project drafting.

| Source ID | Citation | Source type | Abstract-level finding | Usable fact | Supported claim | Citation slot | Risk |
|-----------|----------|-------------|------------------------|-------------|-----------------|---------------|------|
| S01 | Rodkin2011 | article | Seismic regime as avalanche-like relaxations | No complete physical model of preparation | EQ preparation physics remains incomplete | Intro-P1 | low |
| S02 | Cicerone2009 | review | Compilation of EQ precursors | Geophysical/geochemical precursors catalogued | Diverse precursor observations exist | Intro-P1 | low |
| S03 | Conti2021 | critical review | Critical review of ground-based precursors | Precursors need careful validation | Precursor reports ≠ operational forecast | Intro-P1 | low |
| S04 | Geller1997 | critical review | Critique of earthquake prediction | Prediction historically contested | Paper does not claim operational EQ prediction | Intro-P1 / P6 | low |
| S05 | Kirschvink2000 | article | Animal behavior before EQ | Behavioral anomalies; evolutionary sensitivity argument | Animals may respond to preparation cues | Intro-P2 | indirect |
| S06 | Grant2011 | article | Groundwater chemistry and animal effects | Environmental shifts linked to behavior | Biological responses discussed with geochemistry | Intro-P2 | indirect |
| S07 | Yokoi2003 | short report | Mouse circadian activity before Kobe 1995 | Instrumental zoo observation near event date | Precedent for instrumented animal monitoring | Intro-P2 | indirect |
| S08 | Rampin1991 | article | Immobilisation stress → REM rebound in rats | Stress alters REM | REM is stress-sensitive in rats | Intro-P3 | low |
| S09 | Meerlo1997 | article | Social stress → high-intensity sleep | Social stress alters sleep | Different stressors → different sleep shifts | Intro-P3 | low |
| S10 | Palma2000 | article | Cold vs footshock differential sleep effects | Stressor type shapes sleep pattern | REM is not a universal stress meter | Intro-P3 | low |
| S11 | Sanford2003 | article/review | Stress and sleep in rodents | Anxiety/stress linked to sleep structure | Sleep is a plausible state marker | Intro-P3 | low |
| S12 | Vyazovskiy2005 | article | Theta/delta and REM in rats | REM identifiable spectrally; circadian structure | REM measurable from EEG/ECoG | Intro-P4 | low |
| S12b | Fang2010 | article | REM/sleep spectral features in rats | Spectral sleep staging context | Supports REM measurability | Intro-P4 | low |
| S13 | Saevskiy2025 | method | Automated sleep staging algorithm | Local hypnogram pipeline | Continuous hypnograms available at the site | Intro-P4 | project-specific |
| S14 | Hyndman2018 | book | Time-series forecasting principles | Series are autocorrelated | Daily REM summaries are not i.i.d. | Intro-P5 | low |
| S15 | Sinkovec2022 | thesis/ref | Bayesian methods for limited data | Posterior uncertainty with small n | Small samples need explicit uncertainty | Intro-P5 | indirect |
| S16 | Bergmeir2012 | article | CV for time series | Random folds leak temporal structure | Naive CV inflates performance | Intro-P5 | low |
| S17 | Vehtari2017 | article | PSIS-LOO | Model comparison via elpd | Screen configurations by LOO | Intro-P6 | low |
| S18 | old_neuroseismo (user pool) | project reports | REM profile range; ~2–4 day effect window | Classical estimand ≠ onset \(\tau\) | Contrast range-window vs changepoint onset | Intro-P4 | user-archive |
| S19 | carlin1992hierarchical | article | Hierarchical Bayesian changepoint analysis | Bayesian changepoint with uncertainty | Technical task is Bayesian changepoint / regime onset | Intro-P5 / RW-P3 | low |
| S20 | Aminikhanghahi2017 | survey | Survey of time-series changepoint methods | CPD as core time-series mining problem | Positions regime-shift detection for OWAD | RW-P3 | low |
| S21 | Adams2007 | preprint | Bayesian online changepoint / run-length | Online Bayesian regime-shift inference | Contrasts online CPD with our retrospective single-\(\tau\) window | RW-P3 | low |
| S22 | Chandola2009 | survey | Broad anomaly-detection taxonomy | AD as detecting rare/deviant patterns under scarce labels | Anchors anomaly framing beyond CPD alone | RW-P3b / Intro-P4 | low |
| S23 | Gama2014 | survey | Concept-drift adaptation survey | Non-stationary environments need adaptive detection | Seismic / animal context evolves after deployment assumptions | RW-P3b | low |
| S24 | Han2023 | conference (NDSS) | Open-world AD via normality-shift detect/explain/adapt | OWAD namesake line: normality can shift in open world | Venue fit: open-world AD under evolving normality | RW-P3b | low |
| S25 | Faber2024 | journal | Lifelong continual learning for AD | Continual AD must retain past knowledge under drift | Contrast: we do not claim lifelong AD; small-n offline case | RW-P3b / RW-P4 | low |

## Notes

- Do not invent Adams/Fearnhead/Killick-style citations beyond the project bib.
- Classical 2–4 day window: phrase as “classical neuroseismological reports from the same site” until a formal bib key exists.
- Motivation may mention early biosignal indicators; contribution text must not claim operational forecasting.
- Added S20/S21 (2026-08-11) after DOI/arXiv verification for Related Work.
- Added S22–S25 (2026-08-13) via CrossRef DOIs for OWAD venue fit; do **not** claim this manuscript implements Han-style adaptation or Faber lifelong AD.
