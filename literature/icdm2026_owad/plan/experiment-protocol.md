# Experiment protocol (OWAD workshop) — frozen

Источник: `literature/conference_article_ru/plan/experiment-protocol.md` + `run_output_8day_density_safe/`.

## Cohort

- Primary windows after day-mask \(K=6\): \(n=34\)
- Directions: `before`, `after_reversed`
- No random train/test; model comparison via PSIS-LOO on event–animal pairs

## Exploratory grid (completed)

- \(N\in\{12,24\}\), \(\mathrm{ov}\in\{0,0.25,0.5\}\)
- Features: daily/day/night × mean/range (≤3 blocks)
- Mean: student_t, skew_normal; Range: beta, beta_constrained, IIB@0.9, ZOIB
- \(1548\) ok configs; \(366\) per range family with active range
- Artifact: `run_output_8day_density_safe/exhaustive_search_parallel.csv`

## Screening / analysis

- Screen: no \(\mathbb{E}[\tau]\) pile-up at prior floor \(\tau=2\)
- Analysis set: IIB (\(n=366\))
- Controls: plain/BC/ZOIB histograms; mean-only (\(n=84\))

## Confirmatory / sensitivity

- Primary A/B; mask OFF; before_only
- Metrics: \(\mathbb{E}[\tau]\), elpd/(F·E'), R̂, ESS, Pareto-\(\hat k\)

## Within-window day-shuffle null (revision 2026-08-12) — mandatory

**Replaces** the old proxy row-shuffle / calendar date-shuffle for the paper claim.

| Field | Value |
|-------|--------|
| Null | Independently permute **valid** days inside each event window; keep NaN masks in place; do **not** swap whole windows across dates |
| Primary cell | `4f6e4c855d72864d` — `daily:mean`, student_t, \(N=24\), \(\mathrm{ov}=0\), day-mask ON, \(K=6\), \(\tau\in\{2,\ldots,8\}\) |
| Real artefacts | `run_output_8day_density_safe/refit_best_mean_only/rank11_4f6e4c855d72864d/` (`observations.npz`, `trace.zarr` → `tau_probs`) |
| Seeds | `0,1,2` |
| MCMC | Prefer confirmatory: tune 6000 / draws 3000 / chains 4; if too slow, same lighter budget for **both** real comparator and shuffle (document) |
| Outputs | `run_output_*/within_window_shuffle/` — JSON/CSV with mean \(p(\tau=k)\) |
| Paper figure | Variant **A** only (real vs shuffle mean PMF + uniform dashed); `\label{fig:null-within-window}` |
| Claim | Onset tied to **event-aligned day order**; breaks under within-window permutation; not an operational forecast |

**Deprecated for prose:** proxy/date-shuffle narrative and any claim that calendar-date shuffle failed to separate dates.
