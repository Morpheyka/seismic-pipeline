# Seismic Event Pipeline

Pipeline for analysing seismic-event-related changes in sleep/REM profiles from rodent EEG. It uses a scikit-learn-style interface: label generation → hypnogram resolution → REM profile calculation → feature extraction → classification (e.g. logistic regression, SVM) with cross-validation and hyperparameter search.

---

## Task: Auto-Hypnogram from Raw EEG

When hypnograms are missing for some rat/date windows, the pipeline can **compute them from raw `.dat` files** instead of failing or skipping.

- **Input:** Raw EEG `.dat` files (e.g. multi-channel, 250 Hz).
- **Process:** Optional `HypnoCalculatorYt` step: load `.dat` → channel-quality filtering (MLP, classes 4/5) → 3-stage pipeline (artifact GMM → delta/theta thresholds → sleep-stage scoring) → hypnogram array.
- **Output:** Hypnograms are written into the **hypnogram cache** (and optionally to `local_data_root`) so downstream steps (e.g. `REMProfileCalculatorYt`) use them like any other cached hypnogram.

---

## Key Points

### Data sources for `.dat` files

- **Local (mnt):** `/mnt/wd/rat` by default (configurable via `--local-data-root`). Directory layout: `{local_data_root}/{YYYY_MM_DD}/**/*.dat` (filenames contain `rat_id`).
- **S3 fallback:** Bucket `"rat"` (prefix `{YYYY_MM_DD}/`). Used automatically when data is not found on local/mnt.
- **Order:** Try **local first**, then **S3** if missing. Use `--use-s3-dat` to use **only S3** for `.dat` files.

### Caches

- **Hypnogram cache** (`./hypnogram_cache`): Used by REM calculator. Populated from existing files (local/S3) or from **computed** hypnograms via `HypnogramCacheManagerYt.cache_hypnogram_from_data()`.
- **`.dat` file cache** (`./dat_file_cache`): Caches loaded `.dat` data for hypnogram computation (local and S3).

### Pipeline entry point

- **Main script:** `seismic_pipeline_standalone/full_seismic_pipeline_example_window3.py`
- **Enable auto-hypnogram:** run with `--auto-hypnogram` so missing hypnograms are computed from `.dat` (local then S3) before experiments.
- **Hypnogram step:** When `--auto-hypnogram` is set, `HypnoCalculatorYt` is inserted after the label generator and before the REM calculator.

### CLI (selection)

| Option | Description |
|--------|-------------|
| `--auto-hypnogram` | Compute missing hypnograms from `.dat` (mnt then S3 fallback). |
| `--local-data-root` | Root for `.dat` and hypnograms (default: `/mnt/wd/rat`). |
| `--use-s3-dat` | Use only S3 bucket `"rat"` for `.dat` (no local). |
| `--quality-model-path` | Path to channel-quality MLP pickle for auto-hypnogram. |
| `--quality-good-classes` | Comma-separated “good” classes (default: `4,5`). |

### Fallbacks

- If `.dat` is missing or computation fails (e.g. no high-quality channels), the pipeline tries to load an **existing hypnogram** from local then S3 temp bucket; only if both fail is that rat/date treated as missing.

---

## How to run

Use the project’s virtual environment (venv at repo root):

```bash
# From repository root
cd "/path/to/Main project"   # or your repo root
python3 -m venv .venv        # only if .venv doesn't exist
source .venv/bin/activate    # Linux/macOS; on Windows: .venv\Scripts\activate
pip install -r seismic_pipeline_standalone/requirements.txt

cd seismic_pipeline_standalone
# With auto-hypnogram (compute missing from .dat, mnt then S3):
python full_seismic_pipeline_example_window3.py --auto-hypnogram --output-dir ./run_output
# Without auto-hypnogram (only use already-cached/existing hypnograms):
python full_seismic_pipeline_example_window3.py --output-dir ./run_output
```

Configure threading (e.g. `MAX_CORES`, `THREADS_PER_JOB`) at the top of the script if needed.

---

## Citation / conference article

- Release: [`conference-article-2026.08`](https://github.com/Morpheyka/seismic-pipeline/releases/tag/conference-article-2026.08)
- Reproduce: [`REPRODUCE.md`](REPRODUCE.md) (Level A figures from frozen CSV; B/C need data access)
- Cite: [`CITATION.cff`](CITATION.cff)
- Frozen artifacts: `seismic_pipeline_standalone/artifacts/conference_article/`

## Repository layout (main parts)

- `seismic_pipeline_standalone/` – Pipeline package, `requirements.txt` / `requirements-lock.txt`, example runners, scripts.
- `seismic_pipeline_standalone/artifacts/conference_article/` – Frozen CSV + Level A plot scripts.
- `seismic_pipeline_standalone/seismic_pipeline/seismo/` – Hypnogram calculator, cache managers (hypnogram + `.dat`), REM profile calculator, etc.
- Local caches (`hypnogram_cache/`, `dat_file_cache/`, `run_output*`) are not part of the published tree.

---

# Русский перевод

## Пайплайн сейсмических событий

Пайплайн для анализа изменений профилей сна/REM у грызунов, связанных с сейсмическими событиями, по данным ЭЭГ. Интерфейс в стиле scikit-learn: генерация меток → получение гипнограмм → расчёт REM-профиля → извлечение признаков → классификация (логистическая регрессия, SVM) с кросс-валидацией и поиском гиперпараметров.

---

## Задача: авто-гипнограмма из сырых ЭЭГ

Если гипнограммы отсутствуют для части окон крыса/дата, пайплайн может **вычислять их из сырых `.dat` файлов** вместо того, чтобы падать или пропускать даты.

- **Вход:** сырые ЭЭГ `.dat` (многоканальные, 250 Гц).
- **Процесс:** опциональный шаг `HypnoCalculatorYt`: загрузка `.dat` → фильтрация по качеству каналов (MLP, классы 4/5) → 3-стадийный пайплайн (артефакты GMM → пороги дельта/тета → скоринг стадий сна) → массив гипнограммы.
- **Выход:** гипнограммы записываются в **кэш гипнограмм** (и при необходимости в `local_data_root`), чтобы последующие шаги (например `REMProfileCalculatorYt`) использовали их как обычные закэшированные гипнограммы.

---

## Основные моменты

### Источники данных для `.dat` файлов

- **Локально (mnt):** по умолчанию `/mnt/wd/rat` (флаг `--local-data-root`). Структура: `{local_data_root}/{YYYY_MM_DD}/**/*.dat` (в имени файла фигурирует `rat_id`).
- **Резерв S3:** бакет `"rat"` (префикс `{YYYY_MM_DD}/`). Используется автоматически, если данные не найдены локально.
- **Порядок:** сначала **локально**, при отсутствии — **S3**. Флаг `--use-s3-dat` — брать `.dat` **только из S3**.

### Кэши

- **Кэш гипнограмм** (`./hypnogram_cache`): используется калькулятором REM. Заполняется из существующих файлов (локально/S3) или из **вычисленных** гипнограмм через `HypnogramCacheManagerYt.cache_hypnogram_from_data()`.
- **Кэш `.dat`** (`./dat_file_cache`): кэширует загруженные `.dat` для расчёта гипнограмм (локально и S3).

### Точка входа пайплайна

- **Основной скрипт:** `seismic_pipeline_standalone/full_seismic_pipeline_example_window3.py`
- **Включить авто-гипнограмму:** запуск с `--auto-hypnogram`, чтобы недостающие гипнограммы считались из `.dat` (сначала mnt, затем S3) до экспериментов.
- **Шаг гипнограммы:** при `--auto-hypnogram` шаг `HypnoCalculatorYt` вставляется после генератора меток и перед калькулятором REM.

### CLI (основное)

| Опция | Описание |
|--------|-------------|
| `--auto-hypnogram` | Вычислять недостающие гипнограммы из `.dat` (mnt, при отсутствии — S3). |
| `--local-data-root` | Корень для `.dat` и гипнограмм (по умолчанию `/mnt/wd/rat`). |
| `--use-s3-dat` | Брать `.dat` только из S3-бакета `"rat"`. |
| `--quality-model-path` | Путь к MLP pickle качества каналов для авто-гипнограммы. |
| `--quality-good-classes` | «Хорошие» классы через запятую (по умолчанию `4,5`). |

### Резервные варианты

- Если `.dat` нет или расчёт не удался (например, нет каналов хорошего качества), пайплайн пытается загрузить **уже существующую гипнограмму** с локального пути, затем из S3 temp; только при неудаче пара крыса/дата считается отсутствующей.

---

## Запуск

Используйте виртуальное окружение проекта (venv в корне репозитория):

```bash
# Из корня репозитория
cd "/path/to/Main project"   # или ваш корень репо
python3 -m venv .venv        # только если .venv ещё нет
source .venv/bin/activate    # Linux/macOS; в Windows: .venv\Scripts\activate
pip install -r seismic_pipeline_standalone/requirements.txt

cd seismic_pipeline_standalone
# С авто-гипнограммой (вычисление недостающих из .dat, mnt затем S3):
python full_seismic_pipeline_example_window3.py --auto-hypnogram --output-dir ./run_output
# Без авто-гипнограммы (только уже закэшированные/существующие гипнограммы):
python full_seismic_pipeline_example_window3.py --output-dir ./run_output
```

При необходимости настройте потоки (`MAX_CORES`, `THREADS_PER_JOB`) в начале скрипта.

---

## Структура репозитория (основное)

- `seismic_pipeline_standalone/` — пакет пайплайна, `requirements.txt` и пример запуска (`full_seismic_pipeline_example_window3.py`).
- `seismic_pipeline_standalone/seismic_pipeline/seismo/` — калькулятор гипнограмм, менеджеры кэшей (гипнограммы и `.dat`), калькулятор REM-профиля и др.
- `hypnogram_cache/`, `dat_file_cache/` — локальные кэши (часто создаются в корне репозитория при запуске оттуда).
- `copy_plan.md` — план и поток данных для функции авто-гипнограммы.

---

## Bayesian changepoint / conference article

Release [`conference-article-2026.08`](https://github.com/Morpheyka/seismic-pipeline/releases/tag/conference-article-2026.08) — байесовская модель одной точки разладки REM на 8-суточных окнах (density-safe protocol, PSIS-LOO).

- **Reproduce:** [`REPRODUCE.md`](REPRODUCE.md)
- **Article:** [`literature/conference_article_ru/latex/conference_article.pdf`](literature/conference_article_ru/latex/conference_article.pdf)
- **Frozen results:** [`seismic_pipeline_standalone/artifacts/conference_article/`](seismic_pipeline_standalone/artifacts/conference_article/)
- **Main search (article):** `scripts/run_parallel_search_8day_density_safe.py` (IIB/ZOIB grid)
- **Confirmatory:** `scripts/run_density_safe_confirmatory.py`
- **Neg-control:** `scripts/run_density_safe_neg_control.py`

### Байесовский changepoint (кратко)

```bash
cd seismic_pipeline_standalone
pip install -r requirements-lock.txt
python scripts/run_parallel_search_8day_density_safe.py --smoke-only
python ../literature/conference_article_ru/figures/results/plot_e_tau_screening.py
```

---

## Байесовский changepoint / conference article (RU)

См. [`REPRODUCE.md`](REPRODUCE.md) — три уровня воспроизводимости (фигуры за минуты; confirmatory за часы; full search за дни на кластере).
