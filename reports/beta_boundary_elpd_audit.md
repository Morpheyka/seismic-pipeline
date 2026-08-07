# Beta boundary & ELPD audit

**Date:** 2026-08-06  
**Sources:** `run_output_8day_parallel_full/exhaustive_search_parallel.csv`, `top10_refits/`, profile caches, smoke MCMC via `scripts/diagnose_beta_boundary.py`.  
**Artifacts:** `run_output_8day_parallel_full/beta_boundary_diag/beta_boundary_diagnostics.json`

---

## 1. ELPD audit (sign, scale, ranking)

### What the pipeline computes

LOO units are **events** (rows): `changepoint_pointwise_log_lik` sums chunk/day log-lik **within** an event, then ArviZ LOO aggregates over events (`diagnostics.score_changepoint_trace` → `az.loo`).

| Column | Meaning | Scale |
|--------|---------|-------|
| `elpd_loo` | ArviZ expected log pointwise predictive density (sum over events) | log LPD; **can be > 0** |
| `elpd_loo_per_feature_event` | `elpd_loo / (n_features × n_events)` | preferred **ranking** key |
| `loo_ic` | `-2 × elpd_loo` | deviance / IC scale |

Verified on the July CSV: `loo_ic == -2 * elpd_loo` for **100%** of rows. Parallel search already sorts by `elpd_loo_per_feature_event` (eligible first). Exhaustive export now always writes `loo_ic` + optional `*_elpd_legend.md` (`search_export.ELPD_COLUMN_LEGEND`).

### Why ELPD can be positive

Continuous densities are not probabilities. For Beta(α,β) with β < 1 the PDF **spikes as y → 1** and routinely exceeds 1, so log-density is positive. Example under rank-1 posterior means (α≈2.59, β≈0.52):

| y | PDF | log PDF |
|---|-----|---------|
| 0.50 | 0.42 | −0.86 |
| 0.95 | 3.56 | +1.27 |
| 0.99 | 8.24 | +2.11 |

Summing positive log-densities over many observations yields **positive `elpd_loo`**. This is **not** a sign bug; it is ArviZ’s log-score convention. Unit tests: `tests/test_elpd_sign.py`.

### Ranking recommendation

1. Rank models by **`elpd_loo_per_feature_event`** (higher better) so multi-feature models are comparable.  
2. In papers, also report **`loo_ic = −2 × elpd_loo`** or mean log score.  
3. Do **not** treat large positive ELPD under plain Beta as stronger physiology evidence without boundary checks (see §2).

---

## 2. Beta boundary diagnosis (`daily:range`)

### Observation mass near 1 after `/support_upper` (support_upper=2)

| N | overlap | frac(y/2 ≥ 0.9) | frac(y/2 ≤ 0.1) | max(range) |
|---|---------|-----------------|-----------------|------------|
| 48 | 0.00 | **0.277** | 0.000 | 2.0 |
| 48 | 0.25 | 0.284 | 0.000 | 2.0 |
| 48 | 0.50 | 0.295 | 0.000 | 2.0 |
| 24 | 0.00 | 0.246 | 0.000 | 2.0 |

≈ **25–30%** of daily-range values sit in the upper tenth of the Beta unit interval (and hit the clip at 1−ε). No mass near 0.

### Posterior α, β from top-10 refits (plain `range=beta`)

Regime-1 **β means are consistently < 1** (U-shape / right spike):

| rank | likelihoods | β₁ mean | β₂ mean |
|------|-------------|---------|---------|
| 1 | range=beta | **0.52** | 0.96 |
| 2 | range=beta | **0.56** | 1.29 |
| 3 | range=beta | **0.53** | 0.87 |
| 4 | mean=student_t; range=beta | **0.52** | 0.99 |
| 6 | range=beta (N=24) | **0.68** | **0.59** |

Prior Γ(μ=3, σ=1.5) only puts ~5% mass below 1; the **likelihood pulls β below 1** to exploit boundary density.

### Pointwise LL vs y

Under posterior-mean Beta for regime 1: **corr(y, log PDF) ≈ 0.81–0.88**. Observations closer to 1 get systematically higher log-likelihood — the signature of boundary blow-up, not a flat goodness-of-fit gain.

### Full-grid ELPD: plain beta vs IIB (same feature)

From the exhaustive CSV (`daily: range` only):

| N | ov | beta elpd/feat·evt | IIB elpd/feat·evt | Δ | E[τ] beta | E[τ] IIB |
|---|----|--------------------|-------------------|---|-----------|----------|
| 48 | 0 | **6.63** | 4.86 | +1.77 | 6.28 | 7.18 |
| 48 | 0.25 | 6.08 | 4.81 | +1.27 | 5.57 | 7.16 |
| 48 | 0.5 | 6.24 | 4.47 | +1.77 | 7.98 | 7.22 |
| 24 | 0 | 5.82 | 4.41 | +1.41 | 5.70 | 6.61 |

Plain beta’s ELPD lead over IIB is large; τ remains in the ~5.5–8 band either way.

---

## 3. Constrained prior + narrow re-rank (smoke)

### Code path

- Likelihood **`beta_constrained`**: same Beta observation model; shape priors default to `gamma_offset` (Gamma + 1) so **α, β ≥ 1**.  
- Also available: `truncated_gamma` prior dist; preset `PARAMETER_SELECTION_PRESETS["range_beta_constrained"]`.  
- IIB path unchanged. Tests: `tests/test_beta_constrained.py`.

### Smoke MCMC (N=48, ov=0; tune=300, draws=200, 2 chains)

| Model | elpd/feat·evt | E[τ] | frac(β < 1) |
|-------|---------------|------|-------------|
| daily:range + **beta** | **6.62** | 6.23 | β₁: **1.00**, β₂: 0.62 |
| daily:range + **beta_constrained** | **5.18** | 7.29 | **0.00** |
| daily:range + **IIB@0.9** | 4.83 | 7.29 | 0.00 |
| daily:mean+range + student_t + beta_constrained | 5.26 | 7.35 | 0.00 |
| daily:mean+range + student_t + IIB | 5.08 | 7.68 | 0.00 |
| daily:mean+range + normal + beta_constrained | 5.26 | 7.16 | 0.00 |

Constraining shapes (≥1) or using IIB **removes ~1.4 ELPD/feat·evt** from the plain-beta leader while keeping **E[τ] ≈ 6.2–7.7**. After the constraint, range-only / mean+range scores (~5.2) sit near mean-only levels from the full grid (~5.4–5.7), not far above them.

---

## 4. Verdict

| Question | Answer |
|----------|--------|
| Is positive ELPD a bug? | **No** — ArviZ log LPD; Beta PDF > 1 is valid. |
| Is July `daily:range`+plain-beta ELPD leadership an **artifact**? | **Yes, largely.** Mass near 1 + posterior β≪1 + strong corr(y, LL) + ELPD collapse under α,β≥1 / IIB. |
| Is τ an artifact? | **No strong evidence.** τ stays ~6–8 under constrained beta and IIB; onset estimand remains usable. |
| What to report as primary? | Prefer **`daily:mean` / `mean+range`** with student_t/normal, or **`range` with `beta_constrained` / IIB** — not plain beta ELPD winners. Rank by `elpd_loo_per_feature_event`; quote `loo_ic` in text. |

**Bottom line:** treat plain-beta `daily:range` ELPD dominance as a **likelihood boundary artifact**; keep τ≈6.5–7.3 as the robust changepoint finding pending the broader integrity sensitivity table.
