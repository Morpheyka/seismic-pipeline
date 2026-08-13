# Refit diagnostics

| Field | Value |
|-------|-------|
| fingerprint | `00a650a4c85c7621` |
| features | `daily: range` |
| likelihoods | `range=interval_inflated_beta` |
| active feature / lik | `range` / `interval_inflated_beta` |
| n_points / overlap | `12` / `0.0` |
| n_model_events | `33` |
| MCMC | tune=6000, draws=3000, chains=4 |
| tau_map | `7` (conc. 0.475) |

Parameter pairs: `{'mu': None, 'sigma': None, 'alpha': ('alpha_daily_range_1', 'alpha_daily_range_2'), 'beta': ('beta_daily_range_1', 'beta_daily_range_2'), 'pi': ('pi_daily_range_1', 'pi_daily_range_2'), 'nu': None}`

## Figures
| File | Content |
|------|---------|
| `01_traces.png` | MCMC traces |
| `02_params_overlay.png` | Before/after parameter KDEs + P(τ) |
| `03_tau_posterior.png` | Onset posterior |
| `04_likelihood_mean.png` | Density at posterior-mean parameters |
| `05_likelihood_fan.png` | Fan + 90% pdf band (two panels) |
| `06_likelihood_overlay.png` | Median + 90% band, before/after on one axis (no thin curves) |
