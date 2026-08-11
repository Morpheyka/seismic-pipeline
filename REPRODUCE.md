# Воспроизводимость: conference article (density-safe, rev. 3)

Release: [`conference-article-2026.08`](https://github.com/Morpheyka/seismic-pipeline/releases/tag/conference-article-2026.08)

Замороженные результаты: [`seismic_pipeline_standalone/artifacts/conference_article/`](seismic_pipeline_standalone/artifacts/conference_article/)

Исходники статьи (LaTeX/PDF, thesis, черновики) сохранены в git history тега `conference-article-2026.08` и в локальном research-архиве; на публичном `main` оставлен только код и артефакты для цитирования/воспроизведения.

## Окружение

```bash
cd seismic-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r seismic_pipeline_standalone/requirements-lock.txt
```

Кластер: [`scripts/cluster_setup.sh`](scripts/cluster_setup.sh)

## Уровень A — фигуры и числа статьи (минуты, без raw EEG)

Использует committed CSV в `artifacts/conference_article/`.

```bash
cd seismic_pipeline_standalone
python artifacts/conference_article/figures/plot_e_tau_screening.py
python artifacts/conference_article/figures/plot_e_tau_mean.py
```

Сверка: `artifacts/conference_article/figures/data/*.csv` и SVG рядом со скриптами.
Ключевые числа: IIB n=366, E[τ]≈5.48 (см. `fig_iib_main_e_tau_summary.csv`).

## Уровень B — confirmatory + neg-control (часы, n=34)

Требует доступ к hypnogram cache / S3 bucket `rat` или `--local-data-root`.

```bash
cd seismic_pipeline_standalone
python scripts/run_density_safe_confirmatory.py --preset all
python scripts/run_density_safe_neg_control.py --seeds 0,1,2
```

Сверка с `artifacts/conference_article/confirmatory_results.csv` и `neg_control_results.csv`.

## Уровень C — полный exhaustive search (1548 configs, дни)

Entrypoint статьи (IIB/ZOIB/plain Beta grid):

```bash
cd seismic_pipeline_standalone
python scripts/run_parallel_search_8day_density_safe.py --skip-smoke \
  --out-dir ./run_output_8day_density_safe --draws 2000 --tune 4000 --chains 4
```

Post-article эксперимент BC+normal — отдельный скрипт (не смешивать с IIB-grid статьи):

```bash
python scripts/run_parallel_search_8day_density_safe_bc_normal.py --skip-smoke \
  --out-dir ./run_output_8day_density_safe_bc_normal
```

## Ключевые параметры

| Параметр | Значение |
|---|---|
| Окна | 8 суток, `before` / `after_reversed` |
| Events | `FULL_EXHAUSTIVE_EVENTS_8DAY` в `config/changepoint_defaults.py` |
| Day-mask | ON, K=6, артефакты ∪ missing |
| τ prior | {2,…,8}, marginalized NUTS |
| N / overlap | {12,24} × {0, 0.25, 0.5} |
| Range (статья) | plain beta, beta_constrained, IIB@0.9, ZOIB |
| MCMC | BlackJAX NUTS, 4 chains, tune=4000, draws=2000 (full) |

## Статья PDF

PDF статьи был в дереве `literature/` на теге `conference-article-2026.08`
([исторический путь на теге](https://github.com/Morpheyka/seismic-pipeline/blob/conference-article-2026.08/literature/conference_article_ru/latex/conference_article.pdf)).
На очищенном `main` PDF не хранится — используйте release/tag или локальный research-архив.

## Цитирование

См. [`CITATION.cff`](CITATION.cff) или GitHub release `conference-article-2026.08`.
