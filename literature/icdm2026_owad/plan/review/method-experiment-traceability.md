# Method–experiment traceability (OWAD)

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Bayesian single-\(\tau\) anomaly on REM | changepoint + marginalization | exploratory + IIB main | screening / IIB hist | \(\mathbb{E}[\tau]\) stable mid-window under IIB | real (frozen run) |
| Likelihood screening for usable \(\tau\) | Beta-family variants + screen rule | range-family screen | screening table | IIB only family with 0% floor pile-up | real |
| LOO ranks preprocessing, not \(\tau\) interpretability | PSIS-LOO | elpd vs \(\tau\) contrast | elpd medians + top-50 | high elpd ≠ usable \(\tau\) (ZOIB leader) | real |
| Feature geometry / mask sensitivity | profiles + day-mask | Primary A/B, mask OFF, before_only | sensitivity table | mask/stratum shift \(\mathbb{E}[\tau]\) | real |
| Non-causal boundary | — | neg-control shuffle | prose | no sharp null contrast | real; weak causal claim |
