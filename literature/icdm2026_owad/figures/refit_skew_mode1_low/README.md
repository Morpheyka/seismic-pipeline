# Refit diagnostics

| Field | Value |
|-------|-------|
| fingerprint | `036cc0ffcaf8c608` |
| features | `night: mean` |
| likelihoods | `mean=skew_normal` |
| active feature / lik | `mean` / `skew_normal` |
| n_points / overlap | `24` / `0.0` |
| n_model_events | `33` |
| MCMC | tune=6000, draws=3000, chains=4 |
| tau_map | `8` (conc. 0.298) |

Parameter pairs: `{'mu': ('mu_night_mean_1', 'mu_night_mean_2'), 'sigma': ('sigma_night_mean_1', 'sigma_night_mean_2'), 'alpha': ('alpha_night_mean_1', 'alpha_night_mean_2'), 'beta': None, 'pi': None, 'nu': None}`

## Figures
| File | Content |
|------|---------|
| `01_traces.png` | MCMC traces |
| `02_params_overlay.png` | Before/after parameter KDEs + P(τ) |
| `03_tau_posterior.png` | Onset posterior |
| `04_likelihood_mean.png` | Density at posterior-mean parameters |
| `05_likelihood_fan.png` | Fan + 90% pdf band (two panels) |
| `06_likelihood_overlay.png` | Median + 90% band, before/after on one axis (no thin curves) |
