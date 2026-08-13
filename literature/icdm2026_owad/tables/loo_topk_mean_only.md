# Top-10 mean-only by normalized elpd (PSIS-LOO)

Source: `figures/data/mean_only_long.csv` ($n=84$). Ranking metric = `elpd_loo_per_feature_day` = $\overline{\mathrm{elpd}}/(F\cdot E')$.

| # | features | lik | $N$ | ov | $\overline{\mathrm{elpd}}/(F\cdot E')$ | $\mathbb{E}[\tau]$ | max $\hat k$ |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `daily` | Student-$t$ | 24 | 0 | 0.480 | 6.03 | 0.67 |
| 2 | `daily` | skew-normal | 24 | 0 | 0.476 | 6.14 | 1.08 |
| 3 | `daily` | Student-$t$ | 24 | 0.25 | 0.342 | 6.15 | 0.73 |
| 4 | `daily` | skew-normal | 24 | 0.25 | 0.338 | 6.43 | 0.82 |
| 5 | `daily` | skew-normal | 24 | 0.5 | 0.281 | 6.32 | 0.60 |
| 6 | `daily` | Student-$t$ | 24 | 0.5 | 0.279 | 5.57 | 0.65 |
| 7 | `daily+night` | skew-normal | 24 | 0 | 0.276 | 6.38 | 1.03 |
| 8 | `daily+night` | Student-$t$ | 24 | 0 | 0.276 | 5.88 | 0.57 |
| 9 | `daily` | Student-$t$ | 12 | 0 | 0.244 | 5.71 | 0.39 |
| 10 | `daily` | skew-normal | 12 | 0 | 0.243 | 6.22 | 0.78 |
