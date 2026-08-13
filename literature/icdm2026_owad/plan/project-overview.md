# Обзор проекта

## Базовая информация

- **Тип:** workshop paper (ICDM 2026)
- **Воркшоп:** OWAD — Open World Anomaly Detection in Dynamic and Evolving Environments
- **Область:** data mining / changepoint & anomaly-style detection в многомерных временных рядах; домен — нейрофизиология сна (REM) + сейсмический контекст
- **Название:** Bayesian Changepoint Detection in Multivariate REM Time Series Near Seismic Events
- **Дата создания плана:** 2026-08-11
- **Язык:** английский (submission); параллельные русские переводы глав в `chapters/*.ru.md` для автора
- **Вывод:** IEEE LaTeX, ≤8 страниц + 2 extra
- **Ветка:** `workshop/icdm2026-owad` (база: `merge/conference-article-release`)
- **Источник:** `literature/conference_article_ru/`
- **Текущий этап:** черновики глав 01–08 готовы (EN+RU); Abstract и IEEE LaTeX — следующие шаги

## Исследование

### Фон

На станции непрерывно регистрируются ЭКоГ/гипнограммы крыс; классические нейросейсмоотчёты опирались на размах профиля REM в окне ~2–4 суток. Центр данной статьи — байесовский поиск аномалии / смены режима (onset \(\tau\)) в REM-рядах в окрестности сейсмического события. Возможное использование таких аномалий в будущих предиктивных моделях — внешний контекст линии работ, а не сюжет рукописи. Оперативный прогноз здесь не заявляется.

### Цель

Построить байесовскую модель одной точки разладки \(\tau\), выполнить полный перебор предобработки/правдоподобий с ранжированием по PSIS-LOO, оценить устойчивость \(\mathbb{E}[\tau]\) как свидетельство аномалии, связанной с событием, и явно ограничить интерпретацию (не оперативный прогноз, не прямой стресс).

### Методы

- Одна разладка, маргинализация дискретного \(\tau\), NUTS
- Полный перебор конфигураций (\(N\times\mathrm{overlap}\); daily mean/range) + PSIS-LOO
- Primary A/B и integrity — в Discussion, не как единственный сюжет
- Density-safe likelihood для размаха
- Без оперативного прогноза землетрясений и без доказательства «стресса» независимыми биомаркерами

## Позиционирование под OWAD

| Слой | Содержание |
|------|------------|
| Мотивация (лёгкая) | биосигналы REM у сейсмических событий |
| Задача данной статьи | статистически подтвердить аномалию по мозговому REM-сигналу; найти mean-признаки и гиперпараметры, наиболее правдоподобно её описывающие |
| Задача для трека OWAD | changepoint / anomaly-style detection in multivariate time series |
| Вклад | Bayesian \(\tau\) на mean; устойчивые распределения \(\mathbb{E}[\tau]\); ранжирование конфигураций; не bake-off неинформативных range-семейств; не оперативный прогноз |

## Что меняем относительно `conference_article_ru`

- Перевод RU → EN и сжатие под 8+2
- NeurIPS-каркас → IEEE two-column
- Reframing: multivariate changepoint / regime shift на первом плане
- Сейсмика и REM — мотивация и данные, не claim прогноза
- Related Work короче: anomaly/changepoint + biomedical time series + neuroseismo
- Отбор figures/tables под лимит страниц
- Отдельный `plan/` и дерево `literature/icdm2026_owad/`

## Структура глав

1. Introduction
2. Related Work
3. Data
4. Method
5. Experimental Setup
6. Results
7. Discussion & Limitations
8. Conclusion

## Нормы письма

- Язык: английский (workshop)
- Цитирование: bib из проекта + проверка DOI (не выдумывать)
- writing-core: без AI-шаблонов; claims с данными/ограничениями
- Мотивационный хайп не должен превращаться в необоснованный predictive claim
