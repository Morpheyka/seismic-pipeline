# Notes

## Решения брейншторма

- Площадка с финансированием: ICDM 2026 → workshop OWAD
- Идеальных треков BCI / чистый матстат нет; reframing задачи как multivariate time-series changepoint
- Центр статьи: поиск аномалии / разладки в REM, связанной с сейсмическим событием
- **Интерпретационный акцент (2026-08-11):** статистическое подтверждение аномалии по мозговому сигналу (REM/гипнограмма), а не по визуальным наблюдениям; поиск признаков и гиперпараметров, наиболее правдоподобно описывающих аномалию
- **Mean-first:** основной сюжет — распределения \(\mathbb{E}[\tau]\) и конфигурации на mean; для range вторично IIB@0.9 (обычный Beta расходится у 1 при концентрации min–max range), без риторики «априорного исключения семейств»
- Предиктивное моделирование — только внешний контекст линии работ; в теле не раздувать
- Оперативный прогноз не заявляется
- Индекс суток: \(t=8\) = 1 день от события, \(t=1\) = 8 дней; календарная удалённость \(9-t\). В тексте переводить \(\mathbb{E}[\tau]\) в сутки от события, не раздувая «инверсию» как отдельный сюжет
- База кода/текста: `literature/conference_article_ru/` на ветке релиза статьи
- Параллельно с EN-главами писать русские переводы: `chapters/XX_*.ru.md` (для чтения автором; в submission идёт только EN)

## Открытые вопросы на этап написания

- Какой IEEE template/cls взять официально для ICDMW 2026
- Какие 2–4 фигуры оставить в 8 страницах (приоритет: mean-only / mean-context)
- Formalize bib key for classical neuroseismo site reports
- Подтвердить автором: ~~`Passyuk` / `L-ACRD`~~ → **закрыто 2026-08-13:** *Rattus norvegicus*; `L-CARD`; ethics protocol №1 от 1.04.2022; center EN: Scientific Research Technological Center of Neurotechnologies, SFU. Classical 2–4 day cite — нет (статьи в работе).
- ~~Почему IIB `20/33` при `n=34`~~ → **закрыто:** legacy confirmatory `drop_incomplete_events=True` выбросил `R3/2025-01-23/after_reversed` (2 missing days; проходит K=6). Пересчёт density-safe: **20/34**, \(\mathbb{E}[\tau]\approx 6.88\).

## Риски

- Рецензенты OWAD ждут anomaly угол — держать changepoint framing
- Не возвращать длинный bake-off неинформативных range-семейств в Results/Discussion
- **Submission blockers (ревью 2026-08-12):** triple-blind anonymization; ICDM template — **закрыты** IEEE anon path. Остаточный venue-риск OWAD: **смягчён** (2026-08-13) cites Chandola/Gama/Han/Faber + mean title.
- Полный список доебов 12.08: `plan/review_paper_en.md`. Финал 14.08: `plan/review_paper_en_final.md`. **Правки 17.08 закрыты.** **18.08:** легенда Fig. 1 на −3 mm; Fig. 2 без аргументативного title; `nan-padding` убран; Data/Code + `supplement/README.md`. Сабмит только `paper_en.pdf` (8 стр.).

## Решения брейншторма (2026-08-12) — null / claim / viz

Зафиксировано для revision plan `owad_revision_waves`:

- **Обязательный null:** within-window day-shuffle — в каждом окне независимо переставлять **валидные** дни; NaN-маски на местах; **не** мешать целые окна между датами/крысами.
- **Удалить из сюжета статьи:** старый proxy row-shuffle / calendar date-shuffle («shuffle не отделяет даты»).
- **Визуал null:** только **Variant A** — две панели mean \(p(\tau=k)\) real vs shuffle, пунктир \(1/|\mathcal{T}|\) на \(\{2,\ldots,8\}\); макет `figures/null_viz_mockups/variant_A_mean_pmf_real_vs_shuffle.svg`; label `fig:null-within-window`. Variant B не в paper.
- **Primary arm:** fingerprint `4f6e4c855d72864d` — `daily:mean`, Student-\(t\), \(N=24\), \(\mathrm{ov}=0\), day-mask ON, \(K=6\); артефакты `run_output_8day_density_safe/refit_best_mean_only/rank11_4f6e4c855d72864d/` (`observations.npz`, `trace.zarr` / `tau_probs`).
- **Seeds:** `0,1,2`. MCMC: confirmatory (tune 6000 / draws 3000 / chains 4) по возможности; если budget жмёт — один и тот же lighter budget для real и shuffle, явно задокументировать.
- **Claim:** regime-onset pattern зависит от **порядка дней в event-aligned окне**; ломается within-window permutation; не operational forecast.
- **Run status (2026-08-13):** suite 4→3→1→2: синтетика подтверждает мощность null; по 20 seeds mid-window onset **ломается** (peak к краям prior; только 3/20 на τ=7); mean-PMF по 3 seeds был misleading. Prose восстановлен: паттерн чувствителен к порядку суток; календарная каузальность слабая.
