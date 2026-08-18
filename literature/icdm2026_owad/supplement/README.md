# Anonymized derived artifacts (OWAD submission)

This archive contains derived tables and plotting scripts used in the paper. It does not identify authors, institutions, or recording sites. Raw continuous animal recordings are not included.

## Contents

| File | Role |
|---|---|
| `../figures/data/mean_only_long.csv` | Mean-only screening slice (\(n=84\)); source of Table I and Fig. 2 |
| `../figures/data/mean_only_profile_agg.csv` | Aggregates for Fig. 3 (\(N\), overlap) |
| `../figures/data/offline_cpd_baseline.json` | Gaussian two-mean scan on confirmatory daily-mean windows (\(n=33\)) |
| `../tables/loo_topk_mean_only.md` | Top-10 LOO cells |
| `../tables/iib_range_regime_summary.md` | IIB range before/after \(\tau_{\mathrm{MAP}}\) |
| `../tables/student_t_nu_split_vs_shared.md` | Shared vs per-regime \(\nu\) |
| `../figures/plot_fig2_mean_hist.py` | Fig. 2 |
| `../figures/plot_fig3_profile.py` | Fig. 3 |
| `../figures/offline_cpd_baseline.py` | Offline Gaussian scan |
| `../figures/make_loo_topk_table.py` | Table I |

Fingerprints in CSV rows are configuration hashes from the screening grid, not personal identifiers.

A public repository mirroring these derived files will be released upon acceptance.
