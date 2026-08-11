# Table schema (Results)

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|-------|---------|------|---------|-------------|-------------------|
| tab:july-top | Top eligible configs | 5+ | features, lik, N, ov, elpd/FE', E[τ] | `exhaustive_search_parallel.csv` | rerun after likelihood change |
| tab:tau-summary | Stability of τ | Q1 / top-50 | E[τ], sd/median | same CSV | same |
| (opt) tab:beta-boundary | Plain vs constrained/IIB | 3 rows | elpd/FE', E[τ] | beta_boundary audit / smoke | after likelihood freeze |

Figures: existing `q2_tau`-style / hist from July artifacts only if paths cited; no mock as evidence.

PLANNING: after likelihood rework, replace all numeric cells; mark prose `[ДОРАБОТКА]` until then.
