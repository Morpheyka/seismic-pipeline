# Data manifest (figures)

Level A plots live under `seismic_pipeline_standalone/artifacts/conference_article/figures/`.

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Screening E[τ] hist | `data/fig_screening_e_tau_summary.csv`, `fig_screening_e_tau_long.csv` | real | `../exhaustive_search_parallel.csv` | `plot_e_tau_screening.py` | `fig_screening_e_tau_hist.{png,svg}` |
| Screening E[τ] ECDF | same | real | same | same | `fig_screening_e_tau_ecdf.{png,svg}` |
| IIB main E[τ] | `data/fig_iib_main_e_tau_summary.csv` | real | same | same | `fig_iib_main_e_tau_hist.{png,svg}` |
| Mean-only by lik | `data/fig_mean_only_by_lik_summary.csv` | real | same CSV | `plot_e_tau_mean.py` | `fig_mean_only_by_lik.{png,svg}` |
| Mean context 2×2 | — | real | same CSV | `plot_e_tau_mean.py` | `fig_mean_context_e_tau.{png,svg}` |
| Mean+range IIB by lik | — | real | same CSV | `plot_e_tau_mean.py` | `fig_mean_iib_by_lik.{png,svg}` |
