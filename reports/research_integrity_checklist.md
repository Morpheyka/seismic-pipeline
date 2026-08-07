# Чеклист чистоты исследования

**Дата:** 2026-08-06  
**Область:** план §5 (реворк чистоты) — пререгистрация, чувствительность τ, гигиена токсичных профилей, сопоставимость ELPD, негативный контроль.  
**Связанные аудиты:** `[beta_boundary_elpd_audit.md](beta_boundary_elpd_audit.md)`, `[july_recheck_methodology_analysis.md](july_recheck_methodology_analysis.md)`.  
**Smoke препроцессинга:** 10/10 (`tests/test_rem_preprocess_smoke.py` + shape_shift + IIB).

Использовать как гейт перед заявлениями о «победителях» признаков или сравнением LOO между исследованиями. Отмечать пункты по мере появления свидетельств.

---



## 0. Вердикт завершённых аудитов (не переоткрывать)


| Утверждение                                         | Статус                                                                             |
| --------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Положительный `elpd_loo` — баг знака                | **Нет** — ArviZ log LPD; PDF Beta может быть > 1                                   |
| Лидерство July `daily:range` + plain `beta` по ELPD | **В основном артефакт границы** (β≪1, ~25–30% массы около 1 после `/2`)            |
| Constrained α,β≥1 / IIB                             | ELPD падает (~1.4 / feat·evt в smoke); **τ остаётся ~6–7.7**                       |
| Ключ ранжирования                                   | Предпочитать `elpd_loo_per_feature_event`; также сообщать `loo_ic = −2 × elpd_loo` |


**Следствие:** **не** пререгистрировать plain `beta` как основной range-likelihood. Предпочитать `beta_constrained` или `interval_inflated_beta` (IIB@0.9).

---



## 1. Пререгистрация primary-конфигов (до больших выводов)

Primary-анализ фиксируется **до** нарративов ранжирования. Exploratory-сетки допустимы, но в статьях/диссертации цитируются только эти конфиги (плюс таблица чувствительности в §2).

### Primary A — мост к диссертации


| Поле         | Значение                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------ |
| Признаки     | `daily: mean + range`                                                                            |
| Сетка        | `n_points_per_day=24`, `overlap=0.5` (≡ deprecated `W=2,S=1`)                                    |
| Likelihoods  | mean ∈ {`student_t`, `normal`}; range = `beta_constrained` (fallback: IIB@0.9) |
| τ            | маргинализация, `lower=3`, `upper=8` (стандартное окно 8 суток)                                  |
| События      | `FULL_EXHAUSTIVE_EVENTS_8DAY` с протоколом токсичных профилей из §3                              |
| Ранжирование | `elpd_loo_per_feature_event` только среди primary-конфигов                                       |


- [x] Confirmatory MCMC зафиксирован (путь + seeds + draws/tune/chains)
  - **Конфиг:** `daily:mean+range`, mean=`student_t`, range=`beta_constrained`, N=24, ov=0.5
  - **Страта:** `full` (before + after_reversed), 33 exported-события; **2024-09-30 включён**
  - **Бюджет:** tune=500, draws=400, chains=2, `nuts_backend=blackjax` (CPU jaxlib); protocol seed id `20260806` (у `sample_model` нет RNG-seed kw)
  - **Результат (sensitivity run):** E[τ]=**6.285**, τ HDI₆₀ width=**1.342**, elpd/feat·evt=**3.034**, elpd_loo=200.25, loo_ic=−400.50
  - **Alt mean=normal:** E[τ]=6.338, elpd/feat·evt=3.018
  - **Артефакты:** `seismic_pipeline_standalone/run_output_integrity_confirmatory/integrity_sensitivity_full_seed20260806.{json,csv}`; скрипт `scripts/run_integrity_confirmatory.py`
- [x] Сводка Pareto-k приложена; нет тихого исключения событий без страты из §3
  - pareto_k_max=**0.860**, n_over_0.7=**2**: **R3/2024-09-30/before**, **R2/2024-09-30/after_reversed**
  - Incomplete after_reversed (не в likelihood): R2 2024-10-29 (7 missing), R3 2025-01-23 (2), R3 2025-03-14 (8) — `exported=False` в metadata



### Primary B — daily range при density-safe likelihood


| Поле        | Значение                                                                                                  |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| Признаки    | только `daily: range`                                                                                     |
| Сетка       | Confirmatory-тройка: `(N,ov) ∈ {(48,0.0), (48,0.25), (24,0.5)}` — отчитаться по всем трём; не cherry-pick |
| Likelihood  | `beta_constrained` **или** `interval_inflated_beta` **(thr=0.9)** — **не** plain `beta`                   |
| τ / события | как у Primary A                                                                                           |


| (N, ov)   | E[τ]  | HDI₆₀ w | elpd/feat·evt | Заметка        |
| --------- | ----- | ------- | ------------- | -------------- |
| (48, 0.0) | 7.254 | 0.258   | 5.187         | July-подобная  |
| (48, 0.25)| 7.097 | 0.372   | 4.814         |                |
| (24, 0.5) | 5.747 | 2.402   | 4.270         | шире HDI       |


- [x] Plain `beta` допускается только как **диагностический контраст** (boundary-аудит), никогда как заявленный победитель
  - DIAG N=48,ov=0 plain beta: elpd/feat·evt=**6.644** vs constrained **5.187** (тот же builder) — артефакт ELPD подтверждён; E[τ] plain=6.391



### Явно не primary


| Конфиг                               | Роль                                                                |
| ------------------------------------ | ------------------------------------------------------------------- |
| `daily:range` + plain `beta`         | Только диагностика / иллюстрация артефакта                          |
| `shape_shift` / L1–L2 form           | Вторичный / exploratory (July ELPD слабый; τ часто у границы prior) |
| Только half-day без daily-агрегатов  | Exploratory                                                         |
| Cross-builder ELPD vs старый CSV 896 | **Запрещено** (§4)                                                  |


---



## 2. Протокол чувствительности — τ по feature × likelihood × N

**Цель:** отчитаться о **разбросе E[τ]** (и ширине HDI), а не об одном «победителе».  
**Ячейки заполнять после confirmatory-прогонов; пустые — пока не прогнано.** Один и тот же builder, страта событий, бюджет MCMC.

### Сетка протокола

Builder: July `profile_cache/rem_n*_ov*_stage2` + `prepare_model_data` / `build_group_data`. Страта: **full**, 33 события. Бюджет: tune=500, draws=400, chains=2, blackjax. Protocol seed id `20260806`.


| Блок признаков   | Likelihood(s)                 | N   | overlap  | E[τ]  | ширина τ HDI₆₀ | elpd/feat·evt | Заметки                |
| ---------------- | ----------------------------- | --- | -------- | ----- | -------------- | ------------- | ---------------------- |
| daily:mean       | student_t                     | 12  | 0        | 6.157 | 2.144          | 1.679         |                        |
| daily:mean       | student_t                     | 24  | 0        | 6.145 | 1.040          | 3.599         |                        |
| daily:mean       | student_t                     | 24  | 0.5      | 6.043 | 1.835          | 1.807         |                        |
| daily:mean       | student_t                     | 48  | 0        | 6.063 | 2.018          | 5.392         | Pareto: R2/2024-09-30 after |
| daily:mean       | normal                        | 24  | 0.5      | 6.219 | 1.152          | 1.779         | bridge                 |
| daily:range      | **beta_constrained**          | 12  | 0        | 7.031 | 0.046          | 3.167         | узкий HDI у верхней зоны |
| daily:range      | **beta_constrained**          | 24  | 0        | 6.583 | 1.009          | 4.766         |                        |
| daily:range      | **beta_constrained**          | 24  | 0.5      | 5.747 | 2.402          | 4.270         | Primary B              |
| daily:range      | **beta_constrained**          | 48  | 0        | 7.254 | 0.258          | 5.187         | Primary B              |
| daily:range      | **beta_constrained**          | 48  | 0.25     | 7.097 | 0.372          | 4.814         | Primary B              |
| daily:range      | **IIB@0.9**                   | 24  | 0.5      | 7.646 | 0.271          | 4.209         | у верхней границы prior |
| daily:range      | **IIB@0.9**                   | 48  | 0        | 7.016 | 0.431          | 4.843         |                        |
| daily:range      | plain beta                    | 48  | 0        | 6.391 | 0.174          | 6.644         | **только диагностика** |
| daily:mean+range | student_t + beta_constrained  | 24  | 0.5      | 6.285 | 1.342          | 3.034         | **Primary A**          |
| daily:mean+range | student_t + IIB               | 24  | 0.5      | 7.687 | 0.231          | 2.988         | у верхней границы prior |
| daily:mean+range | normal + beta_constrained     | 24  | 0.5      | 6.338 | 1.007          | 3.018         |                        |
| daily:mean+range | student_t + beta_constrained  | 48  | 0        | 7.356 | 0.463          | 5.258         |                        |



### Правила успеха / отчётности

- [x] Сообщить **диапазон** E[τ] по заполненным ячейкам (ожидание ~6–8 по smoke / July)
  - **Без diagnostic plain beta:** E[τ] ∈ **[5.747, 7.687]** (n=16 ячеек).
  - IIB / некоторые range-ячейки тянут к верхней границе prior (~7.6–7.7) с узким HDI — не усреднять вслепую с mid-grid.
- [x] Если E[τ] схлопывается к краям prior (≈3 или ≈8) с плоским posterior — пометить как **неидентифицируемое**, не усреднять вслепую
  - В `full`-страте явного flat@edge нет. **before_only** Primary A: E[τ]=**3.880**, HDI₆₀ w=0.632 — у нижней границы (чувствительность страты, не primary).
- [x] Ранжировать модели внутри **фиксированного** среза feature×N только когда likelihoods сопоставимы по плотности (не plain beta vs student_t как primary-утверждение)
- [x] Путь CSV чувствительности — в Artifacts
  - `seismic_pipeline_standalone/run_output_integrity_confirmatory/integrity_sensitivity_full_seed20260806.csv`

**Уже есть smoke (не полная таблица):** N=48,ov=0 — plain beta E[τ]≈6.23; constrained≈7.29; IIB≈7.29; mean+range constrained/IIB ≈7.2–7.7 (`beta_boundary_elpd_audit.md` §3). **Полная таблица выше — confirmatory 2026-08-06.**

---



## 3. Токсичные профили — исключать / стратифицировать

**Не скрывать** исключения. Всегда указывать страту событий для primary-утверждений.

### 3.1 Высокий Pareto / трудные случаи (July full search)


| Идентичность                                                                     | Проблема                                                                           | Протокол                                                                                       |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **R2 / R3, 2024-09-30** (before + after_reversed)                                | Доминирующие Pareto-исключения (индексы ~6, 23, 24); shape_shift всегда дропает R2 | **Arm чувствительности:** убрать все строки 2024-09-30; сравнить E[τ] и ELPD с полной выборкой |
| R2 2022-11-07 after_reversed; R2 2023-04-11 after_reversed; R1 2025-07-02 before | Вторичный Pareto                                                                   | Оставить в primary; опционально leave-one-out                                                  |


- [x] Primary-таблица указывает, **включён** ли 2024-09-30 или **исключён**
  - Primary A/B (**full**): **включён** (3 строки: R3 before + R2/R3 after_reversed)
- [x] Задокументирована дельта τ при исключении 2024-09-30
  - Primary A full: E[τ]=**6.285**, elpd/fe=3.034
  - Primary A drop_2024_09_30 (n=30): E[τ]=**4.525**, elpd/fe=3.585, pareto_k_max=0.614 (0 over)
  - **ΔE[τ] ≈ −1.76** при дропе; ELPD/fe растёт. Артефакт: `integrity_primary_a_only_drop_2024_09_30_seed20260806.json`



### 3.2 Неполные `after_reversed` (не в 33-event likelihood)


| Событие                      | Проблема                              |
| ---------------------------- | ------------------------------------- |
| R2 2024-10-29 after_reversed | 7 пропущенных дней (`exported=False`) |
| R3 2025-01-23 after_reversed | 2 пропущенных                         |
| R3 2025-03-14 after_reversed | 8 пропущенных                         |


Считать **сбоями доступности данных**, не модели. Не импутировать в LOO.

- [x] Флаги metadata export проверены для confirmatory-прогона
  - `profile_cache/rem_n24_ov0.50_stage2/samples_10days_metadata.csv`: 36 requested → 33 exported; три строки выше `exported=False`



### 3.3 Артефактные дни (form / L1–L2 исследование)

Каноническая карта (также в `scripts/build_seismic_risk_dataset.py` → `ARTIFACT_DAYS_BY_KEY`):


| (rat, date, direction)         | Индексы артефактных дней (0-based в окне) |
| ------------------------------ | ----------------------------------------- |
| R2, 2022-11-07, before         | {0}                                       |
| R2, 2023-05-03, before         | {2}                                       |
| R3, 2023-05-03, before         | {2}                                       |
| R2, 2023-04-21, after_reversed | {3}                                       |


В более раннем form-исследовании также отмечены загрязнённые строки профилей (row 0 day 1; rows 4–5 day 3; row 22 day 4) — см. `reports/rem_profile_artifact_cleaned_study.tex`. Предпочитать автоматические правила артефактов при сборке risk / profile features.

- [ ] Confirmatory-признаки либо маскируют артефактные дни, либо есть чувствительность with/without
  - **Pending:** daily mean/range confirmatory builder **не** маскирует `ARTIFACT_DAYS_BY_KEY` автоматически; with/without sensitivity не прогонялась (нужен отдельный arm). Form-исследование закрывает L1–L2, не Primary A.



### 3.4 Before-only vs full (before + after_reversed)


| Страта                             | Обоснование                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| **before-only**                    | Чище один pre-event changepoint estimand                                         |
| **full** (before + after_reversed) | Как в July exhaustive; after-окна Pareto-тяжёлые и биологически ближе к recovery |


- [x] Primary-утверждение выбирает **одну** страту; другую — как чувствительность
  - **Primary = `full`** (как July). Чувствительность **before_only** Primary A: E[τ]=**3.880**, HDI₆₀ w=0.632, elpd/fe=3.031, n=18; Pareto>0.7: R2/2024-10-29/before, R2/2025-07-02/before (k_max=1.10)
  - before_only тянет τ к нижней границе prior — **не** смешивать с full при заявлениях о E[τ]≈6.3
- [x] Не смешивать страты при сравнении ELPD между текстами

---



## 4. Запрет: ELPD между builders без общего builder


| Разрешено                                                                                       | Запрещено                                                                          |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Сравнивать **τ** (и качественный класс признаков) между 896-config и July                       | Сравнивать абсолютный `elpd_loo` / LOO-IC между старым CSV 896 и July parallel CSV |
| Переэкспортировать обе эры **одним зафиксированным** observation builder, затем сравнивать ELPD | Цитировать «July ELPD ≫ thesis ELPD» как доказательство лучшей физиологии          |
| Ранжировать внутри одного CSV / одного fingerprint builder                                      | Кросс-нормализация ad-hoc константами                                              |


**Почему:** в July переписаны профили (`n_points`+`overlap`, daily-агрегаты). Единицы observation likelihood и число признаков различаются. См. july recheck §8.2–8.3.

- [x] Любая таблица рукописи с ELPD обеих эр использует **общий** пересчёт builder (или опускает ELPD и показывает только τ)
  - Текущие integrity-утверждения используют **только** July profile_cache builder; кросс-эра ELPD не цитируется.

---



## 5. Негативный контроль — shuffle / permute

**Гипотеза при нуле:** если даты событий не несут общей changepoint-структуры, posterior τ должен приближаться к DiscreteUniform на `{3,…,8}` (E[τ]≈5.5 при flat prior), а ELPD должен **ухудшиться** относительно real-date primary.

### Протокол

1. Та же лента событий и primary-конфиг (A или B).
2. Один режим обнуления (задокументировать какой):
  - `shuffle_dates`: внутри каждого `rat_id` переставить `date` между событиями этой крысы (сохраняет животное; ломает календарь). Предпочтительный default.
  - `permute_labels`: глобально перемешать пары `(date, direction)` по событиям (более сильный scramble).
3. Пересобрать признаки / профили для shuffled-событий (те же `N`, overlap, likelihood).
4. Fit с **тем же** бюджетом MCMC, что и real primary confirmatory.
5. ≥3 seeds для формального null; одного seed достаточно для smoke.



### Критерии успеха


| Метрика       | Pass (поддерживает структуру real-date)                                          | Fail (тревога)                    |
| ------------- | -------------------------------------------------------------------------------- | --------------------------------- |
| E[τ]          | Сдвигается к центру prior / уплощается (шире HDI) vs real ~6.5–7.5               | Остаётся остро ~6.5–7 с узким HDI |
| elpd/feat·evt | Явно **ниже**, чем real-date primary                                             | Равен или выше real               |
| Pareto        | Может ухудшиться; не «чинить» дропом событий, если это не зеркалится на real-arm |                                   |




### Скрипт / CLI

```bash
cd seismic_pipeline_standalone

# Протокол + JSON shuffled-событий (без MCMC) — дешёвый smoke
python scripts/run_negative_control_shuffle.py --mode shuffle_dates --seed 0 --dry-run

# Полный null-fit (дорого; тот же builder, что confirmatory)
python scripts/run_negative_control_shuffle.py \
  --mode shuffle_dates --seed 0 --run-mcmc \
  --feature daily:mean+range --n-points 24 --overlap 0.5 \
  --range-likelihood beta_constrained
```

- [x] Stub/скрипт + dry-run smoke задокументированы (`scripts/run_negative_control_shuffle.py`)
- [x] Полный MCMC negative control для Primary A выполнен (путь в дневнике)
  - **Режим:** `shuffle_dates` seed=0; rebuild профилей (`null_mode=shuffle_dates_rebuild`); 34 exported events
  - **Бюджет:** tune=500, draws=400, chains=2 (как Primary A); mean=`student_t`, range=`beta_constrained` + `support_upper=2.0`
  - **Результат:** E[τ]=**5.265** (prior center≈5.5), HDI₆₀ w=**3.625** (vs real 1.34) → **τ-критерий PASS**
  - elpd/feat·evt=**3.051** vs Primary A **3.034** → **ELPD-критерий не выполнен** (почти равен, не ниже)
  - **Итог:** частичный pass — null ломает идентификацию τ; ELPD не ухудшается явно. Нужны ≥3 seeds для формального null (pending).
  - **Артефакты:** `run_output_negative_control/mcmc_result_seed0_shuffle_dates.json`, `events_shuffle_dates_seed0.json`, `profiles_shuffled_seed0_shuffle_dates/`
  - **Баг исправлен:** `_parameter_selection` обязан ставить `support_upper=2.0` (без него ELPD раздувается из-за clip range→[0,1])

---



## 6. Pareto / гигиена отчётности (сквозное)

- [x] Экспорт всегда включает `elpd_loo`, `elpd_loo_per_feature_event`, `loo_ic`
  - Integrity CSV/JSON и July search_export — да; проверено на confirmatory артефактах
- [x] Дропы событий / Pareto-retry перечислены с идентичностями, не только индексами
  - См. §1 Pareto identities; §3.1–3.2 metadata
- [x] Препринт/диссертация: τ устойчив; класс признаков зависит от сетки/likelihood; **нет** уникального победителя plain-beta range
  - Подтверждено DIAG vs constrained; preprint-corea уже согласован
- [x] Лексика согласована со старыми нейросейсмо-отчётами (размах профиля ПС; 12×2ч ≡ N=12,ov=0; окно эффекта 2–4 сут ≠ байесовский onset τ≈6.6)

---



## 7. Справочные артефакты


| Артефакт                                         | Роль                                                        |
| ------------------------------------------------ | ----------------------------------------------------------- |
| `reports/beta_boundary_elpd_audit.md`            | Boundary-артефакт + smoke constrained/IIB; знак ELPD OK     |
| `reports/july_recheck_methodology_analysis.md`   | July vs 896; токсичные профили; эскиз confirmatory-сетки    |
| `reports/rem_profile_artifact_cleaned_study.tex` | Form-исследование артефактных дней                          |
| `scripts/diagnose_beta_boundary.py`              | Density / β<1 / опциональный refit smoke                    |
| `scripts/run_negative_control_shuffle.py`        | Протокол негативного контроля + dry-run / опциональный MCMC |
| `scripts/run_integrity_confirmatory.py`          | Confirmatory Primary A/B + τ sensitivity grid               |
| `run_output_integrity_confirmatory/`             | JSON/CSV чувствительности и страт                           |
| `run_output_negative_control/`                   | Shuffled events + MCMC null                                 |
| `seismic_pipeline/.../changepoint_defaults.py`   | `FULL_EXHAUSTIVE_EVENTS_8DAY`, пресеты likelihood           |
| `RESEARCH_DIARY.md`                              | Журнал сессии по integrity                                  |


---



## Подпись / гейты


| Гейт                                                        | Проверка                             |
| ----------------------------------------------------------- | ------------------------------------ |
| Primary A/B зафиксированы (без plain beta как primary)      | [x]                                  |
| Таблица чувствительности τ заполнена или явно «pending run» | [x] (все ячейки протокола заполнены) |
| Страта токсичных профилей указана                           | [x] (primary=`full`; drop/before arms) |
| Нет cross-builder ELPD в утверждениях                       | [x]                                  |
| Negative-control dry-run OK; full MCMC pending или done     | [x] dry-run / [x] MCMC seed0 (τ pass; ELPD inconclusive; ≥3 seeds pending) |
| Запись в дневнике                                           | [x]                                  |
| Artifact-day mask / with-without sensitivity                | [ ] pending (§3.3)                   |

