## Task Packet
- Scope: уменьшить plate-схему; улучшить читаемость гистограмм; сгенерировать галерею альтернатив вместо `fig:mean-profile` (рис. 3)
- Files to read: `latex/images/tau_model*.tikz`, `figures/data-manifest.md`, `literature/conference_article_ru/figures/data/*.csv`, exhaustive search CSV
- Files allowed to edit: `latex/images/*`, `latex/body_*.tex`, `figures/**`, `plan/progress.md`, `figures/data-manifest.md`
- Required skills: `figures-python`, `experiment-results-planning`
- Evidence/data inputs: mean-only slice из `exhaustive_search_parallel.csv` / long CSV
- Required artifacts: уменьшенный plate; обновлённые hist; галерея кандидатов рис. 3 (PNG+SVG)
- Rejection checks: не использовать mock как «результаты»; подписи осей читаемы в single-column; plate не на всю `\textwidth`
- Validation commands: пересборка `paper_en.pdf` / `paper_ru.pdf`; визуальный просмотр PNG
