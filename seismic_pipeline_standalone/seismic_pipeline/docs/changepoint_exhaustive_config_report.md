# Отчёт: конфигурации для полного перебора changepoint-моделей

Краткий обзор того, что есть в репозитории для **exhaustive / parallel search** байесовских моделей детекции точки изменения (один общий τ, PyMC). Основные источники: `config/changepoint_defaults.py`, `bayesian/priors.py`, `bayesian/parallel_search.py`, `bayesian/changepoint_model.py`.

---

## 1. Что перебирается

Полный перебор — декартово произведение дискретных осей:


| Ось           | Что варьируется                                                                          |
| ------------- | ---------------------------------------------------------------------------------------- |
| REM-профиль   | `n_points_per_day` × `overlap` (+ `rem_stage`)                                           |
| Признаки      | непустые подмножества блоков `(группа, метрика)`, в parallel — с потолком `max_features` |
| Правдоподобия | декартово произведение семейств по активным метрикам                                     |
| τ / scoring   | `tau_threshold`, границы `tau_lower`/`tau_upper` (обычно фиксированы на прогоне)         |


Каждый кандидат — dict с полями:

- `rem_profile_params`
- `n_chunks`
- `feature_selection` — `{group: [metrics…]}`
- `parameter_selection` — `{metric: {likelihood, priors…}}`
- `tau_threshold`

Движки: `exhaustive_model_search` / `_generate_exhaustive_configs` (`search.py`) и `run_parallel_search` + `ParallelSearchConfig` (`parallel_search.py`). Сравнение моделей — elpd LOO (+ фильтры r̂, ESS, divergences, Pareto-k).

**ELPD / LOO columns (CSV):** `elpd_loo` (ArviZ log LPD sum; may be **> 0** for continuous densities such as Beta), `elpd_loo_per_feature_event` = `elpd_loo / (n_features × n_events)` — **preferred ranking key**, `loo_ic = -2 × elpd_loo` (deviance scale). Legend helper: `search_export.elpd_column_legend_markdown()`. See also `reports/beta_boundary_elpd_audit.md`.

---



## 2. Настройки конфигураций



### 2.1 REM-профиль (`REM_PROFILE_CHOICES` / `ParallelSearchConfig`)


| Параметр           | Значения по умолчанию | Ограничения |
| ------------------ | --------------------- | ----------- |
| `n_points_per_day` | `{12, 24, 48}`        | `[4, 96]`   |
| `overlap`          | `{0.0, 0.25, 0.5}`    | `[0, 1)`    |
| `rem_stage`        | `2`                   | —           |


→ **9** профилей в стандартной сетке.

### 2.2 Признаковые блоки

**Группы:** `concat`, `even`, `odd` (также доступны `all`/`daily`/`day`/`night` в других сценариях).

**Метрики:** в сетке обычно `mean`, `range`; дополнительно поддерживаются `std`, `skewness`, `kurtosis`, `shape_shift`.

**ParallelSearchConfig (дефолт):** `max_features=3`, группы `{concat, even, odd}`, метрики `{mean, range}` → 6 блоков, Σₖ₌₁³ C(6,k) = **41** feature-конфиг.

**Режим** `n_chunks`**:**


| `n_chunks_mode`        | Как считается `n_chunks`           |
| ---------------------- | ---------------------------------- |
| `fixed_halfday_chunks` | `n_days * (n_points_per_day // 2)` |
| `window_days`          | `window_days` (типично 8)          |
| `shape_shift`          | `window_days - 1`                  |




### 2.3 Параметры модели τ и MCMC (дефолты parallel search)


| Параметр                    | Дефолт                                |
| --------------------------- | ------------------------------------- |
| `tau_mode`                  | `marginalized` (`discrete` тоже есть) |
| `tau_lower` / `tau_upper`   | `3` / `8`                             |
| `tau_threshold`             | `5.0`                                 |
| `draws` / `tune` / `chains` | `500` / `1000` / `4`                  |
| `nuts_backend`              | `blackjax` (также `pymc`, `numpyro`)  |


Приор на τ: дискретное равномерное на `[tau_lower, tau_upper]` (в marginalized-режиме — равномерный по опоре с маргинализацией).

### 2.4 Готовые пресеты признаков

Из `FEATURE_SELECTION_PRESETS`: например `concat_mean_odd_range`, `concat_mean_range`, `odd_even_mean_range`, варианты с `shape_shift`, `all_metrics`.

---



## 3. Семейства правдоподобия

Реализованы в `build_changepoint_model` (два режима до/после τ):


| Семейство                | Параметры режима | Замечания                                                       |
| ------------------------ | ---------------- | --------------------------------------------------------------- |
| `normal`                 | μ, σ             |                                                                 |
| `student_t`              | μ, σ + общий ν   |                                                                 |
| `lognormal`              | μ, σ (лог-шкала) | clip наблюдений снизу (`eps`)                                   |
| `gamma`                  | α, β             | clip снизу                                                      |
| `beta`                   | α, β             | опционально `support_upper` (для range часто `2.0`); α,β∼Gamma допускают <1 |
| `beta_constrained`       | α, β ≥ 1         | те же Beta observations; приор `gamma_offset` (Gamma+1) — без U-shape у границ |
| `interval_inflated_beta` | π, α, β          | смесь Unif(threshold, 1) и Beta; `threshold` по умолчанию `0.9` |




### Допустимые семейства по метрике (`VALID_LIKELIHOODS`)


| Метрика                 | Разрешено в модели                            |
| ----------------------- | --------------------------------------------- |
| `mean`                  | `student_t`, `lognormal`, `normal`            |
| `range`                 | `beta`, `beta_constrained`, `lognormal`, `interval_inflated_beta` |
| `std`                   | `student_t`, `lognormal`, `gamma`             |
| `skewness` / `kurtosis` | `student_t`                                   |
| `shape_shift`           | `lognormal`, `gamma`                          |




### Что реально в сетке перебора (`LIKELIHOOD_CHOICES_BY_METRIC`)


| Метрика       | Варианты в exhaustive grid                    |
| ------------- | --------------------------------------------- |
| `mean`        | `student_t`, `lognormal`                      |
| `range`       | `beta`, `beta_constrained`, `lognormal`, `interval_inflated_beta` |
| `std`         | `student_t`, `lognormal`, `gamma`             |
| `shape_shift` | `lognormal`, `gamma`                          |


`normal` для mean валиден, но в дефолтную сетку не входит.

---



## 4. Приоры



### 4.1 Дефолты по метрикам (`_default_parameter_selection`)


| Метрика       | Likelihood по умолчанию | `mu_prior`        | `sigma_prior`            | Дополнительно     |
| ------------- | ----------------------- | ----------------- | ------------------------ | ----------------- |
| `mean`        | `student_t`             | Normal(0, 1.5)    | HalfNormal(1.0)          | `nu`: Exp(0.05)+2 |
| `range`       | `lognormal`             | Normal(−0.3, 1.0) | HalfNormal(0.7)          |                   |
| `std`         | `lognormal`             | Normal(−0.7, 1.0) | HalfNormal(0.7)          |                   |
| `skewness`    | `student_t`             | Normal(0, 2.5)    | HalfStudentT(ν=4, σ=1.5) | `nu`: Exp(0.05)+2 |
| `kurtosis`    | `student_t`             | Normal(0, 3.0)    | HalfStudentT(ν=4, σ=2.0) | `nu`: Exp(0.05)+2 |
| `shape_shift` | `lognormal`             | Normal(−2.0, 1.0) | HalfNormal(0.5)          |                   |


У всех метрик по умолчанию `g_prior: {type: "none"}` (без Zellner-масштабирования μ).

### 4.2 Fallback-приоры по семейству (если не переопределены в `parameter_selection`)


| Likelihood                           | Приоры параметров                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `normal` / `student_t` / `lognormal` | μ — из `mu_prior` метрики; σ — HalfNormal (из `sigma_prior` или σ=1); для t: ν ~ Exp+offset |
| `gamma`                              | α, β ~ Exponential(lam=1)                                                                   |
| `beta`                               | α, β ~ Gamma(mu=3, sigma=1.5) — оба могут быть <1 → U-shape / spike у границ |
| `beta_constrained`                   | α, β ~ Gamma_offset(mu=2, sigma=1, offset=1) → поддержка ≥1                 |
| `interval_inflated_beta`             | π ~ Beta(1, 10); α, β ~ Gamma(mu=3, sigma=1); `threshold=0.9`                               |


Пресет `PARAMETER_SELECTION_PRESETS["range_zoib"]` фиксирует ZOIB для range с этими приорами и `eps=1e-6`.
Пресет `PARAMETER_SELECTION_PRESETS["range_beta_constrained"]` — constrained Beta для range с `support_upper=2`.

### 4.3 Фабрика распределений (`_build_prior`)

Поддерживаемые `dist`: `normal`, `halfnormal`, `halfstudentt`, `exponential`, `lognormal`, `exponential_plus`, `beta`, `gamma` (через α/β или μ/σ), `gamma_offset` / `gamma_plus` (Gamma + offset), `truncated_gamma` (Truncated Gamma с `lower`/`upper`).

### 4.4 Опциональные g-priors на μ

Типы: `none` | `unit_information` (g=n) | `zellner_siow` | `hyper_g_n` (Beta(a/2, b/2), по умолчанию a=b=3). Масштабируют σ приора на μ как `sigma * sqrt(g)`. В стандартном exhaustive grid обычно выключены (`none`).

---



## 5. Оценка размера стандартной сетки

При дефолтах parallel search (`9` REM × feature-подмножества ≤3 × likelihood product по активным метрикам) получается **сотни–тысячи** моделей на событие; точный счёт зависит от того, какие метрики входят в каждый feature-конфиг.

Примеры сценариев:

- стандартный `run_parallel_search.py` — mean/range + concat/even/odd;
- `run_parallel_search_shape_shift.py` — только `shape_shift` (~126 моделей при 9 REM);
- `run_parallel_search_8day_events.py` — группы daily/day/night, `window_days`, урезанный набор likelihoods для range.

---



## 6. Эмпирика: какие семейства лучше описывают гистограммы

Диагностика на 33 событиях, REM `n_points=24`, `overlap=0.5`, 8 дней
(`run_output_8day_parallel_full/profile_cache/…`; артефакты в `run_output_likelihood_fit_diagnostics/`).

Сравнение MLE-подгонок (AIC + KS; `powerlaw` отброшен как вырожденный на границах).

### Что видно по форме

| Метрика | Форма гистограммы | Текущая сетка | Проблема |
|---------|-------------------|---------------|----------|
| `mean` (concat/even/odd, day-norm) | платокуртическая (excess kurt ≈ −1.1), слегка скошена | `student_t`, `lognormal` | Normal/t слишком «колоколообразны»; `lognormal` с неверным саппортом |
| `mean` (daily) | почти симметричная | ок | `logistic`/`student_t`/`normal` почти эквивалентны |
| `mean` (day) | левый скос ≈ −0.5 | слабо | нужен `skewnorm` |
| `mean` (night) | шире, намёк на бимодальность | слабо | смесь / `skewnorm` |
| `range` (concat) | сильный правый скос, **~35% массы у 0** (после `/2`) | `beta`, `IIB@0.9`, `lognormal` | IIB бьёт в верхнюю границу, а инфляция **слева** |
| `range` (daily) | ~14% у 1 | `IIB` уместен | — |
| `range` (day/night) | унимодал на (0,1) | `beta` ок | лучше `kumaraswamy` |
| `std` | правый скос от 0 | `lognormal`, `gamma` | `lognormal` сильно проигрывает (`ΔAIC ~1500` на concat) |
| `shape_shift` | умеренный правый скос | `gamma`/`lognormal` | сетка уже адекватна (`gamma` чуть лучше) |
| `skewness` | почти нормальная | `student_t` | достаточно |
| `kurtosis` | правый тяжёлый хвост | `student_t` | нужен `johnsonsu` / `skewnorm` (`ΔAIC ~146`) |

### Рекомендуемое расширение сетки

| Метрика | Добавить в первую очередь | Оставить | Убрать как primary |
|---------|---------------------------|----------|---------------------|
| `mean` | `skewnorm`, `gennorm` (exponential power), `normal_mixture_2` | `student_t`, `normal` | `lognormal` |
| `range` | `kumaraswamy`, **`zero_inflated_beta`**, `logit_normal` | `beta`, `interval_inflated_beta` (для daily) | `lognormal` |
| `std` | `gengamma`, `exponweib` | `gamma` | `lognormal` как единственный |
| `shape_shift` | опционально `gengamma` | `gamma`, `lognormal` | — |
| `kurtosis` | `johnsonsu`, `skewnorm` | `student_t` | — |

Приоритет внедрения в PyMC: **skew-normal**, **2-Gaussian mixture**, **zero-inflated beta**, **Kumaraswamy**, затем **gengamma/exponweib** для `std`.

Подробный JSON: `run_output_likelihood_fit_diagnostics/likelihood_flexibility_recommendation.json`,
оверлеи гистограмм: `…/feature_hist_fit_overlays.png`.

---

## 7. Ключевые файлы

| Файл | Роль |
|------|------|
| `config/changepoint_defaults.py` | REM/feature/likelihood пресеты, события |
| `bayesian/priors.py` | дефолтные приоры, `VALID_LIKELIHOODS`, `_build_prior`, ZOIB |
| `bayesian/changepoint_model.py` | сборка PyMC-модели и сэмплинг |
| `bayesian/parallel_search.py` | `ParallelSearchConfig`, parallel exhaustive |
| `bayesian/search.py` | serial exhaustive + MH по пространству моделей |
| `scripts/run_parallel_search*.py` | entrypoints прогонов |
| `run_output_likelihood_fit_diagnostics/` | эмпирический подбор семейств по данным |


