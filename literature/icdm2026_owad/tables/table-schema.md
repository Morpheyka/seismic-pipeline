# Table schema (OWAD Results)

| Table | Purpose | Rows | Metrics | Data source | Notes |
|---|---|---|---|---|---|
| tab:screening | Range-family \(\tau\) screen | 4 families | mean/median \(\mathbb{E}[\tau]\), % ≤2.05 | density_safe exhaustive | main text |
| tab:elpd-medians | elpd vs family | 4 families | median elpd/(F E') | same | optional compact in prose |
| tab:iib-main | IIB slices | 4 slices | mean/median \(\mathbb{E}[\tau]\), % floor | IIB subset | main text |
| tab:sensitivity | Primary A/B | ~6 rows | \(\mathbb{E}[\tau]\), elpd | confirmatory | compressed |

Figures: reuse `conference_article_ru/figures/results/` (screening hist, mean-only, mean-context, IIB main) — select 2–3 for IEEE page budget.
