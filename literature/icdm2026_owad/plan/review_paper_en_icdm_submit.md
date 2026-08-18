# Финальное ревью `paper_en.pdf` перед подачей

Дата: 2026-08-17. Объект: `literature/icdm2026_owad/latex/paper_en.pdf` (сборка 17 Aug 2026 10:46 MSK).

Критерии формата: [ICDM 2026 Research Track CFP](https://icdm2026.neu.edu.cn/11666/list.htm) (IEEE 2-column, ≤10 стр., triple-blind, reproducibility). Реальная площадка рукописи: **OWAD workshop** (8 стр. контента + 2 на ссылки, тот же IEEE-шаблон, дедлайн 20 Aug 2026).

Вердикт: **для OWAD формат готов к сабмиту** (`paper_en.pdf`, 8 стр., Anonymous). Desk-reject по length / IEEE 2-col / anonymity с ревью 12–14 августа закрыт. Правки 17 августа (фигуры, \(\hat k\), offline scan) закрыли прежние «minor revision» по вёрстке. Остаётся научный риск воркшопа (слабый open-world fit, слабый внешний baseline) и печатная проверка европейских глифов — не блокеры портала.

Сабмитить нужно **`paper_en.pdf`**. Не слать `paper_en_camera.pdf` (имена, 9 стр.).

---

## Общая оценка

**Исследование.** Байесовская одноточечная разладка \(\tau\) на восьмидневных event-aligned средних REM у крыс (\(n=34\)); PSIS-LOO по сетке профилей; within-window day-shuffle как контроль порядка суток; IIB как вторичная проверка размаха; Gaussian two-mean scan как слабый offline baseline.

**Рекомендация.**

- OWAD: слабый, но честно заявленный case study по regime-onset / time-series AD. Non-claim «не continual adaptation» выдерживается до Conclusion. Имеет шанс на workshop, если рецензент принимает domain CPD вместо lifelong AD.
- Research Track ICDM: дедлайн 6 июня 2026 уже прошёл (notifications 16 августа). Даже по формату 8≤10 ок, по вкладу main track не тянет: нет нового алгоритма, \(n=34\), данные не публичны.

**Ключевые плюсы.**

1. Анонимный IEEE two-column, numeric cite, 8 страниц Letter, 10 pt IEEEtran.
2. Mean-first сюжет согласован с title; non-claim доведён до конца.
3. Table I, Fig. 5 (20 seeds), синтетическая мощность null, Data/Code Availability, \(\hat k_{\max}>1\) обсуждён, screening vs confirmatory \(\mathbb{E}[\tau]\) разведены.
4. Правки 17.08 закрыли gallery-title Fig. 3, fingerprints Fig. 4/6, Type 3 на Fig. 5.

**Ключевые минусы (не desk-reject).**

1. Внешний baseline — только retrospective Gaussian two-mean scan (\(n=33\)); явно не BOCPD/PELT/online AD.
2. OWAD-угол заявлен честно, но работа остаётся offline CPD case study.
3. Код «upon acceptance», anonymous repo на ревью нет.
4. Уникальная схема крысы+REM+сейсмо + цитата Saevskiy et al. [14] (соавтор Kosenko в списке) — остаточный риск деанонимизации для специалиста.

---

## Соответствие ICDM 2026 CFP + OWAD

Источники: Research Track CFP; [OWAD CFP](https://sites.google.com/view/icdm-owad-2026-workshop/home); предупреждение портала про Asian/European symbols.

| Требование | Факт | Статус |
|---|---|---|
| IEEE 2-column | `\documentclass[conference]{IEEEtran}`, 10 pt, Letter 612×792 | pass |
| Research Track ≤10 стр. вкл. refs | 8 стр. всего | pass (если бы трек ещё принимал) |
| OWAD regular: ≤8 контента + 2 refs | контент до низа с. 7 (Conclusion + Data/Code); References с с. 7–8, всего 8 | pass |
| Triple-blind | `Anonymous Authors`; Data без станции / ЮФУ / протокола / координат / дат июля | pass с оговоркой |
| Не цитировать свой preprint так, что видна идентичность | self-cite в третьем лице; «our previous work» нет | pass |
| Нет funding / thanks в submission | нет | pass |
| Имя файла не раскрывает авторов | `paper_en.pdf` нейтрально; лучше описательное (`BayesianChangepointREMSeismic.pdf`) | pass, nit |
| Desk reject: anonymity / length / format | формальные триггеры закрыты | pass |
| Reproducibility checklist (Research Track, отдельный PDF Pineau) | в дереве статьи нет; OWAD CFP его не требует, портал CyberChair может всё же спросить | для OWAD не блокер; заполнить, если форма есть |
| Code/data anonymized during review | «will be released upon acceptance» | слабо, не desk-reject |
| Не постить новую версию на arXiv в период ревью | процессный пункт, не PDF | напомнить авторам |
| Duplicate submissions across ICDM workshops | — | не слать параллельно на другой workshop |

Оговорки по анонимности.

- Цитата Saevskiy et al. 2025 [14] + уникальная схема может деанонимизировать специалистам. Для triple-blind это стандартный остаточный риск, не грубый self-cite.
- Camera-копия именная и на 9 страницах — только после acceptance.

---

## PDF print-check: Asian / European symbols

Это ровно то предупреждение, которое цитирует портал. Проверено по `pdffonts`, `pdftotext` и растеризации всех 8 страниц при 160 dpi.

| Проверка | Результат |
|---|---|
| Type 3 fonts | нет ни в `paper_en.pdf`, ни в `latex/images/*.pdf` |
| Все шрифты embedded / subset | да |
| Кириллица в EN PDF | нет (GOST/Rodkin даны латиницей / translit) |
| CJK | нет |
| Европейские акценты | `Benítez` [20], `Šinkovec` [22] есть в текстовом слое; Liberation Serif CID TrueType встроен |
| Blackboard \(\mathbb{E}[\tau]\) на Fig. 2–4 | STIX NonUnicode (PUA в `pdftotext` как ``); на растере читается как \(\mathbb{E}[\tau]\), не как квадратик |
| XeLaTeX + Liberation вместо pdflatex+Times | для review приемлемо; IEEE PDF eXpress на camera может ругаться на набор шрифтов |
| Small-caps IEEEtran | warning `TU/LiberationSerif(0)/m/sc` undefined; заголовки секций всё равно читаются как IEEE caps |
| Overfull `\hbox` | нет |
| Формулы / degree / em-dash | на страницах 1–3 читаются |

Что сделать руками перед загрузкой (5 минут): открыть PDF, пролистать с. 5 (Fig. 2–3, ось \(\mathbb{E}[\tau]\)) и с. 8 (Benítez, Šinkovec). При желании распечатать эти две страницы. Это и есть «print and double check».

Мелочи печати, не блокеры: перенос `electroencephalogram` в [12] может выглядеть как `e-` / `troencephalogram`; в [26] `russian State Standard` с маленькой `r`; в [18] IEEE-разбивка `41 364`.

---

## Рисунки и схемы (все 8 страниц)

Float’ы не падают в References. Размещение: Fig. 1 с. 4; Fig. 2–3 с. 5; Fig. 4–5 с. 6; Fig. 6 с. 7 вместе с Discussion. Fig. 1 (метод) оказывается уже на странице Results — для IEEE нормально, для читателя чуть поздно.

### Fig. 1 — plate

Читается, легенда на месте, dashed box про маргинализацию \(\tau\) цел (формула не обрезана). Два смысловых сдвига относительно текста: узел \(\nu_f\) есть и у skew-normal (у которого \(\nu\) нет); пластина времени \(t=1{\ldots}T\), тогда как окно длины 8 и \(\mathcal{T}=\{2,\ldots,8\}\). Не править в последний день, если нет свободных 15 минут.

### Fig. 2 — гистограммы \(\mathbb{E}[\tau]\)

Ось читается. Gallery/битый глиф с 12–14 августа на печати не воспроизводится. Внутри рисунка остался аргументативный title: «does not pile up at the prior edge». Для IEEE лучше нейтральный title; для сабмита не блокер.

### Fig. 3 — onset vs \(N\), overlap

Gallery-title **снят**. Подпись «mean ± std of \(\mathbb{E}[\tau]\)» соответствует линиям с error bar. Закрыто.

### Fig. 4 / Fig. 6 — диагностики

Fingerprints пайплайна **сняты**. Человеческие titles. Overlay-легенда плотная; в grayscale before/after могут слиться — для review цветного PDF ок.

### Fig. 5 — within-window shuffle

Сюжет верный: real peak на \(\tau=6\)–7 vs shuffle peaks на краях 2 и 8. Type 3 **нет**. Закрыто.

### Table I

Влезает в колонку. Rank 1: daily / Student-\(t\) / \(N=24\) / ov=0 / elpd 0.480 / \(\mathbb{E}[\tau]=6.03\) / \(\hat k=0.67\). Ranks 2 и 7 с \(\hat k_{\max}>1\) в тексте помечены как ordinal-only. Screening 6.03 vs confirmatory 5.74 объяснён бюджетом MCMC. Числа Gaussian scan (\(n=33\), median \(\tau_{\mathrm{ML}}=5.0\), \(10/33\) в \(\{6,7\}\), \(6/33\) = MAP 7) стыкуются с `offline_cpd_baseline.json`.

---

## По разделам

### Abstract / title

Title совпадает с mean-first. Abstract отражает дизайн, null, IIB и non-claim. \(K=6\) для abstract-only читателя всё ещё не определён — nitpick.

### Introduction / Related Work

Дублирование Intro↔RW сохраняется (precursors, REM/stress, non-prediction). Для 8 страниц дорого, логику не ломает. OWAD-абзац честный. Классические нейросейсмоотчёты той же станции (окно 2–4 суток, range) по-прежнему без bib-ключа.

### Data / Method / Setup

Анонимная ветка Data аккуратная. `L-CARD`, *Rattus norvegicus* на месте. Жаргон `nan-padding export` в Setup остался. Method: единый \(\tau\), маргинализация, край prior = «один режим» — согласовано с Results/null. IIB без отдельной методологической ссылки.

Screening MCMC (tune 1000 / draws 500 / 2 chains) на 1548 ячеек лёгкий; сводки \(\hat R\)/ESS по сетке нет.

### Results

Линия mean-only → LOO → confirmatory \(\nu\)-split → shuffle → IIB + Gaussian scan согласована. Detection claim умеренный. \(n=33\) confirmatory vs \(n=34\) screening назван. Обещание Pareto-ID снято.

### Discussion / Conclusion

Ограничения названы. Conclusion близко к Abstract — для workshop допустимо. `visual behavioral proof` выровнен с Abstract (старый `visual ethogram` в EN PDF не всплывает).

---

## Статистика и воспроизводимость

- Единица LOO = event–animal pair при нескольких животных на одни даты: кластерная зависимость не моделируется.
- Null: синтетика показывает мощность; 20 seeds ломают mid-window консенсус.
- Offline baseline слабее, чем ждут ICDM/OWAD-рецензенты, но уже не «ноль сравнения».
- Code/data: upon acceptance. Если портал даёт Anonymous GitHub — положить derived tables.
- Cited keys \(\subseteq\) `references.bib`: 28 ключей, missing 0. Vehtari2024 цитируется.

---

## Язык / writing-core (EN)

Механических `firstly/secondly/it is worth noting/moreover` в рукописи нет. Non-claim повторяется, но не дословным copy-paste. `nan-padding export` — единственный явный pipeline-жаргон.

---

## Что закрыто с 14 августа

Закрыто 17.08: gallery-title Fig. 3; fingerprints Fig. 4/6; Type 3 Fig. 5; `triple-blind` wording; \(\hat k>1\) ordinal; screening vs confirmatory \(\mathbb{E}[\tau]\); Pareto-ID promise снят; Vehtari2024; крошечный offline Gaussian scan; Interpretation subsection убрана.

Не закрыто / не блокер: BOCPD/PELT; classical site-report cite; Intro↔RW дубли; аргументативный title Fig. 2; \(\nu_f\) на plate; `nan-padding`; anonymous code; XeLaTeX vs Times.

---

## Чеклист загрузки (OWAD, до 20 Aug 2026 23:59 AoE)

1. Файл: `paper_en.pdf` (не camera).
2. Имя в портале: описательное, без фамилий.
3. Не публиковать новую arXiv-версию на время ревью (уже лежащий preprint можно не снимать).
4. Если форма просит reproducibility checklist — заполнить отдельно, не вшивать в 8 страниц.
5. Пролистать с. 5 и с. 8 в PDF-ридере (акценты + \(\mathbb{E}[\tau]\)).
6. Не дублировать ту же рукопись на другой ICDM workshop.

Опционально, если есть час: нейтральный title Fig. 2; `nan-padding` → «NaN-masked days retained without imputation»; Anonymous OSF/GitHub на screening tables.

---

## Верификация (команды и факты)

- `pdfinfo paper_en.pdf` → Pages=8, 612×792 pts (letter), PDF 1.5, 2026-08-17 10:46.
- `paper_en.log`: “This is a 10 point document.”; Overfull `\hbox` = 0.
- `pdffonts paper_en.pdf` → Type 3 = 0; emb=yes на всех; CMEX7 uni=no (math, не текст).
- `pdftotext` + grep имён/станции/координат → нет Ponomarev/SFU/Kamchatka/53.06; есть Anonymous Authors; [14] Saevskiy в списке литературы.
- Fingerprints / gallery-title / double-blind → нет.
- Cite keys ⊆ bib → missing [].
- Visual `pdftoppm -r 160` pp.1–8 + crops Fig. 1–5, Table I, refs [20].
- OWAD CFP 8+2 / IEEE 2-col / anonymized / desk reject; ICDM CFP 10 pages / triple-blind.
- `offline_cpd_baseline.json`: n=33, median 5.0, 10/33 in {6,7}, 6/33 MAP 7 — совпадает с текстом.
