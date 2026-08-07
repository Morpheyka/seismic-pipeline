# Experiment protocol (publishable full-rerun, rev. 3)

## Цель

Density-safe полный прогон REM changepoint с единым day-mask (артефакты ∪ missing, K=6), без удаления влиятельных окон. Ранжирование по PSIS-LOO внутри одного builder; устойчивость \(\tau\) и структура рангов.

## Данные / окна

- Окна: 8 суток относительно события; направления `before` и `after_reversed`.
- **Day-mask ON (primary):** для каждого окна `(rat_id, event_date, window_direction)` множество `masked_days` = артефактные дни из канонической карты ∪ дни без гипнограммы (`missing_dates` / metadata).
- Признаки на masked days → **NaN**; min–max нормализация окна — только по валидным дням.
- В changepoint logp: времена с NaN по активным признакам **пропускаются** (без импутации, без укорочения сетки \(\tau\)).
- Окно в выборке ⟺ число валидных дней **≥ K=6**.
- Неполные экспорты: `drop_incomplete_events=False` / nanpad; eligibility по K в builder (не молчаливый дроп всех incomplete).
- Целое окно **не** удаляется из‑за Pareto / «влиятельности»; Pareto-отчёт с ID.
- Нет случайного train/test split; сравнение моделей — LOO по парам событие–крыса.

### Каноническая карта артефактов

| Крыса | Дата события | Направление | Дни 0-based |
|-------|--------------|-------------|-------------|
| R2 | 2022-11-07 | before | `{0}` |
| R2 | 2023-05-03 | before | `{2,3}` |
| R3 | 2023-05-03 | before | `{2}` |
| R2 | 2023-04-21 | after_reversed | `{1,2}` (не `{3}`) |

### Incomplete / missing (K=6)

| Окно | Решение |
|------|---------|
| R3 2025-01-23 after_reversed | включить (~6 валидных) |
| R2 2024-10-29 after_reversed | исключить (<K) |
| R3 2025-03-14 after_reversed | исключить (<K) |

Ожидаемый n primary: ~34 окна (уточнить после re-export).

## Сетка (замороженный дизайн)

| Решение | Значение |
|---------|----------|
| Признаки | `daily` / `day` / `night` × `mean` / `range` (комбо ≤3) — полусутки в **main** |
| N | `{12, 24}` только (без N=48) |
| overlap | `{0, 0.25, 0.5}` → 6 профильных ячеек |
| Mean lik | `student_t` + `skew_normal` (`normal` вне сетки) |
| Range lik | `beta_constrained` + `interval_inflated_beta@0.9` + `zero_inflated_beta` (настоящий ZOIB) |
| Plain `beta` | DIAG только, вне eligible |
| \(\tau\) prior | `{2,…,8}`, marginalized NUTS |
| Primary stratum | `full` + **mask ON** |
| Sensitivity | (1) mask OFF на той же когорте окон; (2) `before_only` + mask ON |
| Primary A | `daily:mean+range`, student_t + beta_constrained, N=24, ov=0.5 |
| Primary B | `daily:range` на `(24,0)` / `(24,0.25)` / `(24,0.5)` с beta_constrained и ZOIB |
| Ranking | `elpd_loo_per_feature_event` внутри одного builder |
| Влиятельные окна | не дропать |

Оценка объёма exploratory: ~500–900 конфигов.

## Метрики / таблицы

- **T1** Primary A/B (mask ON): E[τ], mode, HDI₆₀, elpd±SE, elpd/(F·E'), R̂, ESS, Pareto-k + ID, n, n_masked_days
- **T2** sensitivity: mask ON vs OFF; full vs before_only
- **T3** exploratory visit rates; диапазон E[τ]; без unique winner
- **T4** DIAG plain beta vs constrained / IIB / ZOIB
- Primary: \(\overline{\mathrm{elpd}}/(F E')\), \(\mathbb{E}[\tau]\), HDI/Pareto-\(k\)
- Не использовать LOO-IC как отдельную метрику в тексте статьи
- Сравнивать elpd только внутри одного builder / одного прогона

## Negative control / PPC

- Neg-control: `shuffle_dates` seeds `{0,1,2}` на Primary A (тот же mask)
- PPC + DIAG + ключевые фигуры (p(τ), stability, Pareto, boundary, профиль с маской)

## Флаги публикации

- [ ] Mask ON primary; K=6; missing и артефакты в Methods
- [ ] after_reversed маска `{1,2}` (не канон `{3}`); R3 2025-01-23 в когорте
- [ ] Main ranking без plain beta; ZOIB + skew_normal в тексте
- [ ] Нет N=48; τ∈{2…8}; Pareto без silent drop
- [ ] T2 mask on/off и before_only; null ≥3 seeds
- [ ] PDF без `[ДОРАБОТКА]` по числам; нет EQ-прогноза / «стресс доказан»
- В тексте статьи **не** упоминать неопубликованную диссертацию / сетку 896
