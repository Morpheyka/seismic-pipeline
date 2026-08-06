# Перепроверка июля 2026: методология n_points / overlap — анализ и рекомендации

**Дата:** 2026-08-06  
**Область:** повторный анализ переписанного в июле пайплайна changepoint (фиксированные окна → `n_points_per_day` + `overlap`), сравнение с исследованием 896 конфигураций W/S и синтез с более ранними нейросейсмо-манускриптами.

---

## 1. Краткое резюме

Перепроверка в июле на переписанном коде **не отменяет** основное научное утверждение диссертации/препринта: общий REM changepoint остаётся около **\(\mathbb{E}[\tau]\approx 6.5\)–\(6.7\) дней** до события. При новой параметризации профилей и группировке признаков **меняется то, какие признаки побеждают в LOO-ранжировании**.

| Исследование | Сетка | Класс признаков на верхних позициях | \(\mathbb{E}[\tau]\) (лучший квартиль / top-50) |
|-------|------|-------------------|-----------------------------------------------|
| 896-config (диссертация) | \(W\in\{2,3,6\}\), \(S\in\{1,2\}\) | **`concat:mean`** (+ student_t / lognormal) | \(\approx 6.64\)–\(6.73\) |
| July parallel full | \(N\in\{12,24,48\}\), overlap \(\in\{0,0.25,0.5\}\) | **`daily:range`** (+ beta); затем daily mean+range | \(\approx 6.55\) (top-50), \(6.70\) (Q1) |
| July shape_shift-only | та же \(N\times\)overlap | odd/even `shape_shift` (gamma) | \(\approx 5.15\) (слабее, часто у нижней границы prior) |

**Итог для дизайна исследования:** сохранить байесовский вывод по \(\tau\); **перецентрировать признаки на daily range (и daily mean+range)**; рассматривать **`shape_shift` как вторичный/исследовательский**; оставить **student_t / normal + beta** основными likelihood; планировать **skew-normal / zero-inflated beta** как следующий шаг по prior–likelihood. Профили около **2024-09-30 (R2/R3)** и несколько неполных окон **after_reversed** — главные случаи слабой детектируемости / высоким Pareto.

---

## 2. Смена методологии: фиксированные W/S → n_points + overlap

### 2.1 Старая параметризация (896-config / диссертация)

Суточные REM-профили строились из скользящих окон:

- `window_size_hours` \(W \in \{2,3,6\}\)
- `step_size_hours` \(S \in \{1,2\}\)
- `rem_stage = 2`

Блоки признаков: `concat` / `even` / `odd` × `{mean, range}` с общим chunked \(\tau\).

### 2.2 Новая параметризация (переписанный пайплайн июля)

Канонические параметры профиля (см. `config/changepoint_defaults.py`, `docs/changepoint_exhaustive_config_report.md`):

| Параметр | Сетка | Границы |
|-----------|------|--------|
| `n_points_per_day` | `{12, 24, 48}` | `[4, 96]` |
| `overlap` | `{0.0, 0.25, 0.5}` | `[0, 1)` |
| `rem_stage` | `2` | — |

Устаревший `(W,S)` всё ещё принимается, но **deprecated** и конвертируется:

\[
n_{\text{points}} = 24/S,\qquad
\text{overlap} = 1 - S/W \quad (W > S).
\]

| Старый \((W,S)\) | Эквивалент \((n,\text{overlap})\) |
|---------------|-------------------------------------|
| (2, 1) — фаворит диссертации | (24, 0.50) |
| (2, 2) | (12, 0.00) |
| (3, 1) | (24, 0.67) — **вне** новой стандартной сетки |
| (6, 1) | (24, 0.83) — **вне** новой стандартной сетки |

Таким образом, июльская сетка **не является строгим надмножеством** старой W/S-сетки: она убирает окна 3–6 ч с высоким overlap, но **добавляет** более высокую дискретизацию \(N=48\) и явную ось без перекрытия / частичного перекрытия.

### 2.3 Почему переписывание важно научно

1. **Идентифицируемость:** \(N\) и overlap разделяют «сколько суточных отсчётов» и «сколько временного сглаживания», вместо смешения в \((W,S)\).
2. **Сопоставимость с классической нейросейсмо-работой:** в более ранних манускриптах Saevskiy et al. использовались **12 неперекрывающихся 2-часовых бинов** \(\equiv (N=12,\ \text{overlap}=0)\).
3. **Новая онтология признаков в основном июльском запуске:** агрегаты `daily` / `day` / `night` (один скаляр на сутки или полусутки), а не только серии chunk concat/even/odd — ближе к классическому пайплайну «суточный размах REM-профиля».

Якоря в коде:

- `seismic_pipeline_standalone/seismic_pipeline/config/changepoint_defaults.py` (`REM_PROFILE_CHOICES`, `validate_rem_profile_params`)
- `seismic_pipeline_standalone/seismic_pipeline/seismo/rem_export.py`
- `seismic_pipeline_standalone/seismic_pipeline/features/rem_chunk_features.py`
- Точки входа: `scripts/run_parallel_search.py`, `run_parallel_search_8day_events.py`, `run_parallel_search_shape_shift.py`

---

## 3. Данные июльских перепроверок (конкретные артефакты)

### 3.1 Основной июльский exhaustive search (`run_output_8day_parallel_full`)

**Артефакт:** `seismic_pipeline_standalone/run_output_8day_parallel_full/exhaustive_search_parallel.csv`

| Величина | Значение |
|----------|-------|
| Завершённые конфигурации | **1224** (все `status=ok`) |
| Участвуют в ранжировании | **1079** |
| REM-сетка | \(N\in\{12,24,48\}\times\) overlap \(\in\{0,0.25,0.5\}\) |
| Набор событий | 36 перечисленных окон; **33 экспортированных** полных 8-дневных профилей |
| Скоринг | PSIS-LOO ELPD; ранжирование по `elpd_loo_per_feature` |

**Top-15 eligible (сокращённо):**

| Rank | Features | Likelihoods | \(N\) | ov | ELPD/feat | \(\mathbb{E}[\tau]\) |
|------|----------|-------------|------|-----|-----------|----------------------|
| 1 | daily: range | range=beta | 48 | 0.00 | 6.63 | 6.28 |
| 2 | daily: range | range=beta | 48 | 0.50 | 6.24 | 7.98 |
| 3 | daily: range | range=beta | 48 | 0.25 | 6.08 | 5.57 |
| 4 | daily: mean, range | student_t + beta | 48 | 0.00 | 6.03 | 6.47 |
| 5 | daily: mean, range | normal + beta | 48 | 0.00 | 5.99 | 6.80 |
| 6 | daily: range | range=beta | 24 | 0.00 | 5.82 | 5.70 |

**Структура top-50:**

- Признаки: `daily: mean, range` (28), `day: range` (15), `daily: range` (14), `daily: mean` (8); night присутствует, но вторичен.
- Likelihood: `range=beta` (34), `mean=normal` (22), `mean=student_t` (17), `range=interval_inflated_beta` (11).
- Разрешение: **\(N=48,\ \text{ov}=0\)** доминирует (26/50); затем (48, 0.25), (48, 0.50). Грубое \(N=12\) почти не появляется в верхней части.
- \(\mathbb{E}[\tau]\) top-50: **mean 6.55, median 6.62**; eligible Q1 (лучший квартиль ELPD): **6.70 ± 0.93**.

**Более длинные MCMC refits** (`top10_refits/`, tune=6000, draws=3000) подтверждают тот же костяк ранжирования. Пример posterior means для \(\tau\):

| Refit rank | Config | \(\tau\) mean (refit) |
|------------|--------|------------------------|
| 1 | daily:range, beta, N=48, ov=0 | 6.28 |
| 2 | daily:range, beta, N=48, ov=0.5 | 7.98 (близко к верхней границе — осторожность) |
| 4 | daily:mean+range, student_t+beta, N=48, ov=0 | 6.52 |
| 5 | daily:mean+range, normal+beta, N=48, ov=0 | 6.78 |

Источник: `…/top10_refits/refit_summary.json`, `…/rank*/posterior_summary.csv`.

### 3.2 Абляция shape-shift (`run_output_8day_parallel_shape_shift`)

**Артефакт:** `run_output_8day_parallel_shape_shift/exhaustive_search_parallel.csv` (126 configs).

- Лучшие модели: **`odd:shape_shift` / even+odd**, likelihood **gamma**, в основном **\(N=12\)**.
- ELPD per feature **отрицательный** (\(\approx -6.4\) у лучших) — не конкурирует с daily range/mean моделями (положительный ELPD/feat \(\approx 5\)–\(6.6\)).
- \(\mathbb{E}[\tau]\) top-20 \(\approx 5.15\) с частым MAP у нижней границы (3) — **слабый / нестабильный сигнал changepoint**.
- **Каждый** завершённый запуск удаляет event index **5 = R2 2024-09-30** (Pareto retry) — систематический outlier для shape-признаков.

Вывод: shape/L1–L2 динамика интересна описательно (см. §3.3), но **не должна быть первичной** для байесовского поиска с общим \(\tau\).

### 3.3 Исследование формы REM-профиля (13 Jul 2026)

**Артефакт:** `reports/rem_profile_artifact_cleaned_study.tex`

- Протокол уже использует **\(N=24\), overlap=0.50** (и проверяет все 9 ячеек \(N\times\)overlap).
- Описательный рост late-vs-early L1/L2 **стабилен по всем 9 сеткам профилей**.
- Формальные тесты (дни 6–8 vs 4–5; расстояния внутри блока) в основном **не значимы**; один исследовательский сигнал L1 на полупрофиле (\(p_{\text{one}}=0.0449\)).
- Явно исключённые артефакты: row 0 day 1; rows 4–5 day 3; row 22 day 4.
- Замечание автора: полупрофиль ≠ проверенный фотопериод до проверки маппинга.

### 3.4 Эмпирические диагностики likelihood / prior

**Артефакты:**

- `seismic_pipeline_standalone/run_output_likelihood_fit_diagnostics/likelihood_flexibility_recommendation.json`
- `seismic_pipeline_standalone/run_output_8day_parallel_full/likelihood_recommendation.json`
- сводка в `docs/changepoint_exhaustive_config_report.md` §6

Ключевые эмпирические несоответствия с *текущей* exhaustive-сеткой:

| Метрика | Наблюдаемая форма | Проблема сетки | Предпочтительно |
|--------|----------------|------------|--------|
| mean (concat/even/odd, day-norm) | плосковершинная, слабый skew | student_t/normal недообучают; lognormal — неверная support | skewnorm, gennorm, 2-Gaussian mixture |
| mean (daily) | почти симметричная | OK | normal ≈ student_t |
| range (concat) | ~35% массы около 0 | IIB@0.9 попадает в *верхнюю* границу | **zero-inflated beta** / left inflation |
| range (daily) | часть массы около 1 | IIB уместен | beta + IIB |
| shape_shift | правый skew | уже OK | **gamma** ≳ lognormal |
| std | правый skew от 0 | lognormal теряет большой ΔAIC | gengamma / exponweib |

Операционная рекомендация, уже использованная в июльском full search (`likelihood_recommendation.json`): mean ∈ `{normal, student_t}`; range ∈ `{beta, interval_inflated_beta}` с `support_upper=2` и scaling `/2`.

### 3.5 Бенчмарк старого vs переписанного кода (инженерный, не научный)

**Артефакты:** `run_output_old_vs_current_exhaustive/benchmark_summary_final.json`, `selected_configs_from_june.json`

- Выбранные победители июня всё ещё были **W/S `concat:mean`** configs.
- Текущие blackjax-запуски медленнее и не сравнимы численно 1:1 (другой observation builder / шкала ELPD) — трактовать как **стресс-тест реализации**, а не научное противоречие.

### 3.6 Линия predictive risk (только контекст)

`reports/seismic_risk_predictive_study.tex` + `run_output_risk_model/`: датасет собран (54 строк), **CV не завершён**. Ортогонально к changepoint-перепроверке, но часть июльского workstream.

---

## 4. Выводы из `literature/old_neuroseismoreports`

Три Word-манускрипты (допребайесовская / параллельная нейросейсмо-линия):

1. **`manuscript_REM_quakes.docx`** — «Изменение характеристик парадоксального сна…»  
2. **`manuscript_REM_forecast.docx`** — правило краткосрочного прогноза из изменений REM  
3. **`Био_сейсмопрогнозирование_25_09_2025_САИ.docx`** — сводка биосейсмического мониторинга на уровне станции  

### 4.1 Доктрина признаков в классической линии

- Суточный REM **profile** из **неперекрывающихся 2-часовых окон → 12 points/day** (\(\equiv N=12,\ \text{overlap}=0\)).
- Первичный скаляр: **range (размах)** этого профиля — разность max−min, намеренно смешивающая экстремумы светлого/тёмного периода.
- Профили min–max нормируются по окрестности события перед объединением.

Это **совпадает с июльским байесовским победителем (`daily:range`)** ближе, чем с фаворитом диссертации `concat:mean`.

### 4.2 Утверждения о сроках

- Значимые аномалии размаха **до ~2–4 дней до** и **~2–4 дней после** сильных событий (интенсивность >2).
- Сильнейшее послесобытийное искажение часто на дни **+1…+3**; пресобытийный эффект наиболее ясен на **−2, −1** в effect-size plots.
- Манускрипт прогноза: пороговое правило по изменениям REM за **~3 дня** до события; расширённый набор событий (12 vs 6).

### 4.3 Согласие / расхождение с байесовским \(\tau\approx 6.6\)

| Классическая нейросейсмо | Байесовский changepoint (896 + July) |
|-----------------------|-----------------------------------|
| Эффект виден за 2–4 д до события | Changepoint **~6.5–6.7 д** до события |
| Акцент на **range** | Диссертация акцентировала **concat:mean**; июльская перепроверка возвращает **daily:range** |
| Неперекрывающиеся 2-ч бины (N=12) | July предпочитает **N=48** для daily aggregates; N=12 всё ещё совместим с классикой |
| Сильная **послесобытийная** реакция | Модели используют `before` и `after_reversed` окна; after-окна дают много Pareto-outliers |

**Интерпретация (обоснованная, не доказанная):** классические тесты детектируют момент, когда *эффект становится большим относительно тихих контролей*; байесовская модель с одним \(\tau\) ставит *начало нового режима* раньше. Это совместимо, если отклонение нарастает за несколько дней. Это **не** один и тот же estimand.

---

## 5. Priors, которые стоит сохранить (с обоснованием)

Defaults в `bayesian/priors.py` (`_default_parameter_selection`, IIB helpers) и `PARAMETER_SELECTION_PRESETS`.

### 5.1 Оставить как основные research priors

| Цель | Prior / likelihood | Почему |
|--------|--------------------|-----|
| Общий \(\tau\) | Discrete uniform на \(\{3,\ldots,8\}\) (marginalized) | Совпадает с обоими исследованиями; posterior mass остаётся внутри для хороших моделей; thesis Q1 \(\approx 6.6\) |
| **daily / concat mean** location | \(\mu\sim\mathrm{Normal}(0,1.5)\), \(\sigma\sim\mathrm{HalfNormal}(1)\); lik. **student_t** или **normal**; \(\nu\sim\mathrm{Exp}(0.05)+2\) | July top ranks; daily mean почти симметрична; student_t полезен при contamination |
| **daily range** | Beta likelihood на range/2; \(\alpha,\beta\sim\mathrm{Gamma}(\mu=3,\sigma=1.5)\) (code default для beta) | Доминирует в July LOO; совпадает с классическим «размах»; bounded support |
| **IIB для daily range** | \(\pi\sim\mathrm{Beta}(1,10)\), \(\alpha,\beta\sim\mathrm{Gamma}(\mu=3,\sigma=1)\), threshold \(0.9\) | Конкурентен в top-50 при насыщении daily range у верхней границы; держать как *вторичный* |
| **shape_shift** (исследовательский) | \(\mu\sim\mathrm{Normal}(-2,1)\), \(\sigma\sim\mathrm{HalfNormal}(0.5)\); lik. **gamma** | Эмпирически gamma ≳ lognormal; priors соответствуют малым положительным сдвигам |
| g-prior на \(\mu\) | **`none`** для exhaustive ranking | Избегает смешения при сравнении моделей; `hyper_g_n` / Zellner–Siow только в sensitivity checks |

### 5.2 Понизить приоритет или перепроектировать

| Спецификация | Действие | Почему |
|------|--------|-----|
| **lognormal для day-normalized mean** | Убрать из primary grid | Неверная support / плохий MLE fit; July grid уже убрал для daily run |
| **lognormal для range** | Не использовать как primary | Bounded [0, 2]; beta family побеждает |
| **IIB@0.9 для concat range** | Не считать default | Масса около **0**, не 1 — нужна **zero-inflated / left-boundary** inflation |
| Очень широкие priors на kurtosis/skewness | Оставить только если метрики вернутся в сетку | Не в winning July feature sets |

### 5.3 Следующие шаги prior–likelihood (приоритет реализации)

Из MLE diagnostics (July):

1. **Skew-normal** (mean, day/night mean)  
2. **Zero-inflated beta** (concat/even/odd range)  
3. **gennorm / 2-Gaussian mixture** (platykurtic concat mean)  
4. Опционально **Kumaraswamy** для interior (0,1) day/night range  

---

## 6. Признаки, которые важны (с обоснованием)

### 6.1 Первичные (продолжать)

1. **`daily:range`** — July LOO #1–3; связь с классическим размахом; β likelihood.  
2. **`daily:mean + daily:range`** — стабильный top-10; \(\tau\) posterior ~6.5–6.8 на refits.  
3. **`daily:mean` alone** — всё ещё сильный; немного более ранний/узкий \(\tau\) (~6.0) чем range-only.  
4. **Сетка профилей:** предпочитать **`N=48`, overlap \(\in\{0,0.25\}\)`** для daily aggregates; сохранить **`N=24`, ov=0.5`** как мост с диссертацией (`W=2,S=1`).

### 6.2 Вторичные

5. **`day:range`** (полупрофиль) — частый компаньон в top-50; трактовать как зонд циркадной структуры, не в одиночку (half-day-only configs начинаются ~rank 97).  
6. **Legacy `concat:mean`** под chunked \(\tau\) — всё ещё чемпион 896-study; полезен для **непрерывности репликации** и утверждений препринта.  
7. **Night metrics** — в top-100, но не доминируют; полезны для робастности, не для headline.

### 6.3 Слабые / малоинформативные для shared-\(\tau\) search

8. **`shape_shift` / L1–L2 form distances** — отрицательная шкала ELPD; \(\tau\) нестабилен; только описательно.  
9. **Высшие моменты** (skewness, kurtosis) — не нужны для текущих winning models.  
10. **`std` отдельно от range** на 2-point chunks — в основном избыточен с range.

### 6.4 Практическая рецептура признаков

Для следующего confirmatory run:

```text
features: {daily: [range]} and {daily: [mean, range]}
N × overlap: (48,0.0), (48,0.25), (24,0.5)   # last = thesis bridge
likelihoods: mean ∈ {student_t, normal}; range ∈ {beta, interval_inflated_beta}
tau: marginalized, lower=3, upper=8
```

Опциональная continuity arm: `concat:mean` с \((N=24,\text{ov}=0.5)\) или deprecated `(W=2,S=1)`.

---

## 7. Слабо детектируемые профили (с обоснованием)

### 7.1 Высокий Pareto / repeatedly removed events (July full search)

Model indices на **33 экспортированных** профилях (`profile_cache/rem_n24_ov0.50_stage2/samples_10days_metadata.csv`):

| Model idx | Removal count (across configs) | Identity |
|-----------|--------------------------------|----------|
| **23** | **403** | **R2, 2024-09-30, after_reversed** |
| **6** | **298** | **R3, 2024-09-30, before** |
| 18 | 38 | R2, 2022-11-07, after_reversed |
| 24 | 36 (also high residual influence) | **R3, 2024-09-30, after_reversed** |
| 21 | 28 | R2, 2023-04-11, after_reversed |
| 10 | 26 | R1, 2025-07-02, before |

**Кластерный диагноз:** событие **2024-09-30** (R2 и R3, before *и* after) — наиболее ясный систематический трудный случай. After-окна непропорционально вносят Pareto-k — согласуется с сильным *послесобытийным* искажением сна в классических работах (режим может не быть чистым single pre-event \(\tau\)).

### 7.2 Неполные / неэкспортированные after windows

Не вошли в 33-event likelihood (metadata `exported=False`):

- R2 2024-10-29 after_reversed (7 missing days)  
- R3 2025-01-23 after_reversed (2 missing)  
- R3 2025-03-14 after_reversed (8 missing)  

Это **сбои доступности данных**, не сбои модели — но они искажают покрытие after_reversed.

### 7.3 Трудные случаи shape-shift

- **R2 2024-09-30** удалён во **всех 126** shape_shift configs (index 5 в том event list).  
- Вторичные удаления: R4/R3 2025-07-02 after windows, R3 2025-01-23, R2 2023-04-21.

### 7.4 Дни с артефактами (form study)

Из `rem_profile_artifact_cleaned_study.tex`: row 0 day 1; rows 4–5 day 3; row 22 day 4. До появления автоматических правил артефактов эти дни **загрязняют признаки и reference profiles**.

### 7.5 Формы признаков с провалом детекции

- **Shape-only features** (shape_shift, L1/L2 to early reference) — слабая формальная evidence; плохий байесовский ELPD.  
- **Half-day-only models без daily aggregates** — низкий LOO rank.  
- **after_reversed** как зеркальный «pre-event» changepoint — часто Pareto-heavy; биологически другой процесс (recovery / post-stress).

---

## 8. Сравнение с выводами 896-config

### 8.1 Согласия (сохранить)

1. **Общий \(\tau\) около 6.5–6.7 дней** переживает переписывание методологии и другую онтологию признаков.  
2. **Короткие, сравнительно мелкие профили** побеждают грубое/сильное сглаживание (старое: \(W=2\); новое: \(N=48\) или thesis-bridge \(N=24\)).  
3. **Range остаётся научно центральным**, когда восстановлен классический daily aggregate — хотя 896 ранжировал concat:mean первым.  
4. **Pareto-k hygiene важна**; top models могут быть чистыми (\(k<0.7\)), но несколько событий доминируют в retries.  
5. **Не операционный прогноз землетрясений** — только REM-associated state change (неизменённое замечание).

### 8.2 Противоречия / сдвиги (документировать явно)

| Тема | 896 / диссертация | July recheck | Чтение |
|-------|--------------|--------------|---------|
| Winning feature | `concat:mean` | `daily:range` | Разные конструкторы признаков; оба OK для \(\tau\), **не взаимозаменяемы для LOO claims** |
| Preferred likelihood (mean) | student_t **and** lognormal | normal / student_t; lognormal demoted | MLE + July grid согласны: lognormal — плохий primary для normalized mean |
| Preferred resolution | \(W=2,S=1\) \(\to\) (24, 0.5) | (48, 0.0) для daily | Более мелкая дискретизация помогает daily scalars; (24, 0.5) остаётся continuity bridge |
| Timing narrative | \(\tau\approx 6.6\) d | то же | Классические 2–4 d papers измеряют **более поздний/больший** эффект, не тот же onset parameter |
| Shape features | не в 896 grid | tested; **lose** | Не заменяют mean/range |

### 8.3 Импликации для текста препринта / диссертации

- Сохранить \(\tau\approx 6.6\) как робастное утверждение.  
- Смягчить любую импликацию, что **`concat:mean` уникально предпочтительен**; указать, что при daily aggregates **range (β)** ведёт LOO, согласно более ранним нейросейсмо-манускриптам.  
- Отметить, что July использовал **переписанный observation pipeline**; абсолютные ELPD не сравнимы 1:1 с June CSV fingerprints без frozen builder.

---

## 9. Открытые вопросы / следующие эксперименты

1. **Frozen confirmatory grid** (рецепт в §6.4) с более длинным MCMC (как в top10 refits) и pre-registered event exclusions для sensitivity 2024-09-30.  
2. Реализовать **zero-inflated beta** + **skew-normal**; переранжировать малую сетку.  
3. Сопоставить half-profile indices с **истинными light/dark** фазами перед интерпретацией day/night biology.  
4. Раздельные модели для **before** vs **after_reversed** (или hierarchical animal/event effects) — after windows выглядят как другая generative story.  
5. Завершить **risk-model CV** (`run_bayesian_risk_cv.py`) с non-fallback quiet controls.  
6. Prospective holdout на новых событиях с фиксированным `daily:range`, \(N=48\), ov=0, beta + \(\tau\) prior как выше.  
7. Согласовать классические 2–4 d effect-size curves с байесовским \(\tau\) через **growth / two-changepoint** или gradual-transition model.

---

## 10. Индекс артефактов

| Path | Role |
|------|------|
| `final_results/final_no_pareto.csv` | 896-config W/S results (диссертация) |
| `seismic_pipeline_standalone/run_output_8day_parallel_full/exhaustive_search_parallel.csv` | July main LOO table (1224) |
| `…/top10_refits/` | Longer MCMC refits + posterior summaries |
| `…/likelihood_recommendation.json` | Likelihoods used in July full search |
| `run_output_8day_parallel_shape_shift/exhaustive_search_parallel.csv` | Shape-shift ablation (126) |
| `seismic_pipeline_standalone/run_output_likelihood_fit_diagnostics/` | MLE family diagnostics |
| `reports/rem_profile_artifact_cleaned_study.tex` | Form / L1–L2 study (13 Jul 2026) |
| `reports/seismic_risk_predictive_study.tex` | 24 h risk protocol (CV pending) |
| `run_output_old_vs_current_exhaustive/` | Old vs new code benchmark |
| `literature/old_neuroseismoreports/` | Classical REM–quake manuscripts |
| `seismic_pipeline_standalone/seismic_pipeline/docs/changepoint_exhaustive_config_report.md` | Config/prior/likelihood map |
| `seismic_pipeline_standalone/seismic_pipeline/bayesian/priors.py` | Prior factory |
| `seismic_pipeline_standalone/seismic_pipeline/config/changepoint_defaults.py` | REM grid + W/S deprecation |
| `RESEARCH_STATUS_2026-08-06.md` | High-level project status |
| `literature/latex/mmcs-sfedu_thesis7/preprint_seismic_stress.tex` | Preprint claims |

---

## 11. Вердикт в одном абзаце

Июльское переписывание корректно отказывается от смешанных \((W,S)\) окон и переходит на **`n_points` + `overlap`**, и при этом протоколе данные предпочитают **суточный REM range (beta)** при **мелкой дискретизации (\(N=48\))**, при этом восстанавливая тот же **~\(6.6\)-дневный** changepoint, что в 896-config **`concat:mean`** study. Классические нейросейсмо-отчёты уже указывали на **range** и более короткие **2–4 дневные** контрасты; байесовская модель оценивает **более раннее начало режима**. Research priors должны оставаться сконцентрированными на **Normal/HalfNormal + student_t/normal для mean**, **Gamma-parameterized beta (и careful IIB) для range** и **uniform discrete \(\tau\)**; **shape_shift** и профили **2024-09-30 / incomplete after** трактовать как известные failure modes, а не как тихие загрязнители.
