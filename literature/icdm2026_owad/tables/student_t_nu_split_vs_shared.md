# Student-\(t\): shared \(\nu\) vs \(\nu\) per regime (existing artifacts only)

No new MCMC. Sources:

| Run | Path | fingerprint |
|-----|------|-------------|
| shared \(\nu\) | `run_output_8day_density_safe/refit_best_mean_only/rank11_4f6e4c855d72864d/` | `4f6e4c855d72864d` |
| \(\nu\) split | `run_output_8day_density_safe/refit_student_t_nu_split/rank11_4f6e4c855d72864d_nu_split/` | `4f6e4c855d72864d_nu_split` |

Both: `daily:mean`, Student-\(t\), \(N=24\), \(\mathrm{ov}=0\), \(n_{\mathrm{events}}=33\), tune=6000 / draws=3000 / chains=4. \(\tau_{\mathrm{MAP}}=7\) in both.

| Quantity | shared \(\nu\) | \(\nu\) split |
|----------|---------------:|--------------:|
| \(\mathbb{E}[\tau]\) (posterior mean) | 5.74 | 6.01 |
| \(\tau_{\mathrm{MAP}}\) concentration | 0.309 | 0.408 |
| \(\nu\) | shared \(\mathbb{E}[\nu]\approx 20.7\) (median \(\approx 15.4\); from `trace.zarr`) | \(\nu_1\approx 29.8\), \(\nu_2\approx 13.0\) |
| \(\mu_1,\mu_2\) | \(-0.396\), \(-0.408\) | \(-0.396\), \(-0.409\) |
| \(\sigma_1,\sigma_2\) | \(0.137\), \(0.154\) | \(0.138\), \(0.144\) |
| \(\sigma_2-\sigma_1\) | \(\approx 0.017\) | \(\approx 0.006\) |

Reading for Results/Discussion:
- Bulk location (\(\mu\)) almost unchanged; the \(\sigma\)-gap shrinks under per-regime \(\nu\).
- After-regime degrees of freedom are smaller (\(\nu_2<\nu_1\)): heavier tails after the MAP onset.
- Confirmatory cell only — not part of the primary LOO ranking.
