# Gallery: кандидаты вместо рис. 3 (`fig:mean-profile`)

Все графики по **реальным** mean-only ячейкам (`n=84`) из `exhaustive_search_parallel.csv`.
Выберите букву (A–O); текущий вставленный в PDF пока остаётся старым `fig_mean_by_profile` до вашего выбора.

| ID | File | Idea |
|---|---|---|
| A | `A_lines_improved` | улучшенный baseline: линии onset vs overlap |
| B | `B_heatmap_etau` | теплокарта mean $\mathbb{E}[\tau]$ по $N\times$overlap |
| C | `C_heatmap_elpd` | теплокарта mean elpd_loo |
| D | `D_scatter_elpd_etau` | scatter elpd vs onset (size∝overlap) |
| E | `E_box_by_profile` | boxplot по ячейкам профиля |
| F | `F_violin_days_from_event` | violin календарной дистанции $9-\mathbb{E}[\tau]$ |
| G | `G_paired_student_vs_skew` | парный сдвиг Student-$t$ vs skew-normal |
| H | `H_forest_profile_cells` | forest plot ячеек профиля |
| I | `I_box_by_feature_family` | onset по семейству признаков (daily/day/night) |
| J | `J_small_multi_hist` | small multiples гистограмм |
| K | `K_hdi_vs_etau` | ширина HDI vs onset |
| L | `L_ecdf_by_N` | ECDF onset по $N$ |
| M | `M_top_elpd_bubbles` | top-24 по elpd |
| N | `N_grouped_bars` | grouped bars |
| O | `O_slope_chart` | slope chart |

Файлы: `figures/gallery_fig3/*.png` (+ `.svg`).

Также: `alt_hist_stacked_column` — вертикальный вариант гистограмм для одноколоночной вёрстки.