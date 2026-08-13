# Refit diagnostics

| Field | Value |
|-------|-------|
| fingerprint | `4f6e4c855d72864d_nu_split` |
| features | `daily: mean` |
| likelihoods | `mean=student_t;nu_per_regime` |
| active feature / lik | `mean` / `student_t` |
| n_points / overlap | `24` / `0.0` |
| n_model_events | `33` |
| MCMC | tune=6000, draws=3000, chains=4 |
| tau_map | `7` (conc. 0.408) |

Parameter pairs: `{'mu': ('mu_daily_mean_1', 'mu_daily_mean_2'), 'sigma': ('sigma_daily_mean_1', 'sigma_daily_mean_2'), 'alpha': None, 'beta': None, 'pi': None, 'nu': ('nu_daily_mean_1', 'nu_daily_mean_2')}`

## Figures
| File | Content |
|------|---------|
| `01_traces.png` | MCMC traces |
| `02_params_overlay.png` | Before/after parameter KDEs + P(τ) |
| `03_tau_posterior.png` | Onset posterior |
| `04_likelihood_mean.png` | Density at posterior-mean parameters |
| `05_likelihood_fan.png` | Fan + 90% pdf band (two panels) |
| `06_likelihood_overlay.png` | Median + 90% band, before/after on one axis (no thin curves) |
