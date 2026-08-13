# Notes

## Решения брейншторма

- Площадка с финансированием: ICDM 2026 → workshop OWAD
- Идеальных треков BCI / чистый матстат нет; reframing задачи как multivariate time-series changepoint
- Центр статьи: поиск аномалии / разладки в REM, связанной с сейсмическим событием
- **Интерпретационный акцент (2026-08-11):** статистическое подтверждение аномалии по мозговому сигналу (REM/гипнограмма), а не по визуальным наблюдениям; поиск признаков и гиперпараметров, наиболее правдоподобно описывающих аномалию
- **Mean-first:** основной сюжет — распределения \(\mathbb{E}[\tau]\) и конфигурации на mean; неинформативные range-семейства (известная a priori несостоятельность граничных Beta/ZOIB и т.п.) не раздувать и не делать центром статьи
- Предиктивное моделирование — только внешний контекст линии работ; в теле не раздувать
- Оперативный прогноз не заявляется
- Индекс суток: \(t=8\) = 1 день от события, \(t=1\) = 8 дней; календарная удалённость \(9-t\). В тексте переводить \(\mathbb{E}[\tau]\) в сутки от события, не раздувая «инверсию» как отдельный сюжет
- База кода/текста: `literature/conference_article_ru/` на ветке релиза статьи
- Параллельно с EN-главами писать русские переводы: `chapters/XX_*.ru.md` (для чтения автором; в submission идёт только EN)

## Открытые вопросы на этап написания

- Какой IEEE template/cls взять официально для ICDMW 2026
- Какие 2–4 фигуры оставить в 8 страницах (приоритет: mean-only / mean-context)
- Formalize bib key for classical neuroseismo site reports
- Подтвердить автором: `Passyuk` (линия крыс) и `L-ACRD` vs `L-CARD`; аффилиации соавторов (KB GS RAS?)
- Почему IIB event-wise счётчик `20/33` при `n=34`

## Риски

- Рецензенты OWAD ждут anomaly угол — держать changepoint framing
- Не возвращать длинный bake-off неинформативных range-семейств в Results/Discussion
- **Submission blockers (ревью 2026-08-12):** triple-blind anonymization; ICDM template; desk-reject policy на format
- Полный список доебов: `plan/review_paper_en.md`

## Решения брейншторма (2026-08-12) — null / claim / viz

Зафиксировано для revision plan `owad_revision_waves`:

- **Обязательный null:** within-window day-shuffle — в каждом окне независимо переставлять **валидные** дни; NaN-маски на местах; **не** мешать целые окна между датами/крысами.
- **Удалить из сюжета статьи:** старый proxy row-shuffle / calendar date-shuffle («shuffle не отделяет даты»).
- **Визуал null:** только **Variant A** — две панели mean \(p(\tau=k)\) real vs shuffle, пунктир \(1/|\mathcal{T}|\) на \(\{2,\ldots,8\}\); макет `figures/null_viz_mockups/variant_A_mean_pmf_real_vs_shuffle.svg`; label `fig:null-within-window`. Variant B не в paper.
- **Primary arm:** fingerprint `4f6e4c855d72864d` — `daily:mean`, Student-\(t\), \(N=24\), \(\mathrm{ov}=0\), day-mask ON, \(K=6\); артефакты `run_output_8day_density_safe/refit_best_mean_only/rank11_4f6e4c855d72864d/` (`observations.npz`, `trace.zarr` / `tau_probs`).
- **Seeds:** `0,1,2`. MCMC: confirmatory (tune 6000 / draws 3000 / chains 4) по возможности; если budget жмёт — один и тот же lighter budget для real и shuffle, явно задокументировать.
- **Claim:** regime-onset pattern зависит от **порядка дней в event-aligned окне**; ломается within-window permutation; не operational forecast.
- **Run status (2026-08-13):** suite 4→3→1→2: синтетика подтверждает мощность null; по 20 seeds mid-window onset **ломается** (peak к краям prior; только 3/20 на τ=7); mean-PMF по 3 seeds был misleading. Prose восстановлен: паттерн чувствителен к порядку суток; календарная каузальность слабая.
