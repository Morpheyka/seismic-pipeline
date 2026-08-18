# Figure data manifest (OWAD)

| Figure | Asset | Role | Keep for 8+2 |
|---|---|---|---|
| `fig:tau-model` | `latex/images/tau_model.tikz` / `tau_model_en.tikz` | plate-схема модели в Method | yes |
| `fig:mean-only` | `latex/images/fig_mean_only_by_lik.{png,pdf,svg}` | гистограммы \(\mathbb{E}[\tau]\) mean-only; нейтральный title (`plot_fig2_mean_hist.py`) | yes |
| `fig:mean-profile` | `latex/images/fig_mean_by_profile.{png,pdf,svg}` | mean \(\mathbb{E}[\tau]\) vs overlap for \(N=12,24\) | **dropped from paper 18.08** |
| gallery A–O | `figures/gallery_fig3/*` | кандидаты вместо рис. 3 | choose one |
| data | `figures/data/mean_only_long.csv`, `mean_only_profile_agg.csv` | real mean-only slice | — |
| refit student_t mean | `figures/refit_best_mean_only/` | `4f6e4c855d72864d` daily:mean student_t | internal |
| refit skew (eligible) | `figures/refit_skew_normal_mean/` | `0da9bbc60553996e` daily:mean skew, E[τ]≈6.39 | internal |
| refit skew mode1 low | `figures/refit_skew_mode1_low/` | `036cc0ffcaf8c608` night:mean skew, lower E[τ] mode (~5.4) | internal |
| refit skew mode2 high | `figures/refit_skew_mode2_high/` | `65515c6970fa7603` daily:mean skew, upper E[τ] mode (~6.4) | internal |
| refit IIB range t=0.9 | `figures/refit_iib_range/` | `00a650a4c85c7621` daily:range IIB@0.9; obs /2 | internal |
| refit IIB range t=0.85 | `figures/refit_iib_range_t085/` | same setup, threshold=0.85 | internal |
| refit script | `figures/refit_best_mean_only/plot_refit_diagnostics.py` | student_t / skew_normal / IIB diagnostics | — |
| refit student_t nu_split | `figures/refit_student_t_nu_split/` | same as best mean student_t + `nu_per_regime=true` | internal |
| `fig:diag-iib` | `latex/images/fig_diag_iib_{params,overlay}.{png,pdf,svg}` | IIB range confirmatory; human title, no fingerprint | yes |
| `fig:diag-mean-nu` | `latex/images/fig_diag_mean_nu_{params,overlay}.{png,pdf,svg}` | mean Student-\(t\) \(\nu\)-split; human title, no fingerprint | yes |
| `fig:null-within-window` | `latex/images/fig_null_within_window.{png,pdf,svg}` | real PMF + 20-seed peak histogram; Type 1/TrueType, no Type 3 | yes |
| offline CPD | `figures/data/offline_cpd_baseline.json` | Gaussian two-mean scan, \(n=33\) confirmatory mean windows | prose |
| IIB regime table | `tables/iib_range_regime_summary.md` | before/after \(\tau_{\mathrm{MAP}}=7\) on \(y=\mathrm{range}/2\) | prose |
| \(\nu\)-split table | `tables/student_t_nu_split_vs_shared.md` | shared vs existing \(\nu\)-split posterior summary | prose |
