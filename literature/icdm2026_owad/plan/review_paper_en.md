# Ревью `paper_en.pdf` (ICDM 2026 OWAD) — список замечаний

Дата ревью: 2026-08-12. Объект: `literature/icdm2026_owad/latex/paper_en.pdf` (8 стр., letter, twocolumn). Вердикт рецензента: **major revision / риск desk reject** из‑за формата и анонимизации; по содержанию — интересный domain case, но слабая привязка к OWAD и слабый контроль.

---

## 0. Критично для submission (desk reject)

- Нет triple-blind анонимизации: на титуле имена и аффилиация; OWAD явно требует anonymized IEEE ICDM submission.
- Не использован официальный шаблон ICDM 2026 / IEEE conference: сейчас `\documentclass[10pt,twocolumn]{article}` + самодельный `geometry`, а не ICDM template.
- Дата `\maketitle` печатает «August 12, 2026» — для workshop submission лишняя и выглядит как черновик.
- Ключевое слово / позиционирование «open-world anomaly detection» есть в Keywords, но в Related Work почти нет OWAD-литературы (continual AD, concept drift, open-set / novelty). Риск «wrong venue / weak fit».

---

## 1. Научные слабости (major)

- Title обещает *Multivariate* REM time series, а основной сюжет — скалярные mean (иногда несколько блоков с общим \(\tau\)). Либо сузить title, либо явно показать multivariate joint likelihood / shared-\(\tau\) multi-feature как главный результат.
- Главный negative control (proxy date-shuffle) **не отделяет** реальные даты от shuffled; при этом Detection claim формулируется довольно уверенно («anomaly is present»). Нужно либо усилить контроль, либо смягчить detection claim до «coherent mid-window onset under retained specs» без сильного «anomaly confirmation».
- Нет сравнения с классическими CPD / AD baselines (BOCPD/Adams уже цитируется, но не сравнивается; CUSUM, PELT, offline Bayesian alternatives, simple ML AD). Для ICDM/OWAD это ожидаемо.
- LOO «ranks» заявлен в Abstract/Results, но **нет таблицы** top-k конфигураций, \(\Delta\mathrm{elpd}\), SE, Pareto-\(\hat k\). Утверждение о ранжировании не верифицируемо из PDF.
- Классические нейросейсмоотчёты той же станции (эффект 2–4 суток, range) — центральная мотивация — **без цитаты**. Рецензент попросит источник или уберёт claim.
- `Sinkovec2022` (PhD thesis) тянет на роль опоры small-sample Bayesian uncertainty — слабая и косвенная ссылка для ключевого методологического абзаца.
- `Adams2007` цитируется как journal-style «Bayesian Online Changepoint Detection», но в `.bib` это только arXiv preprint без DOI/venue — выглядит незавершённо.
- Выборка \(n=34\) окон / 14 дат / несколько животных на одних датах: зависимость event–animal pairs не моделируется (нет hierarchical / clustered LOO). Independence assumption для PSIS-LOO observation unit спорна.
- `20/33` окон в IIB secondary check при когорте \(n=34\): отсутствует объяснение, куда делось одно окно.
- Screening MCMC (tune 1000, draws 500, 2 chains) очень лёгкий для 1548 конфигураций; нет сводки \(\hat R\) / ESS failures по сетке.
- Range families «excluded a priori» звучит как post-hoc narrative control: без пререгистрации / appendix с дисквалифицированными семействами рецензент может заподозрить cherry-picking.
- Ethics: approval mentioned, но нет номера протокола / даты.
- Нет Data/Code availability — для reproducible Bayesian grid это минус на ICDM workshop.

---

## 2. Структура и риторика

- Introduction и Related Work сильно дублируют друг друга (precursors → animals → REM/stress → CPD → non-claims). Related Work не добавляет достаточно новой карты литературы.
- Non-claim «not operational earthquake prediction» повторяется слишком часто (Abstract, Intro, Related Work, Discussion, Conclusion) — выглядит defensive boilerplate.
- Фраза про exclusion range families повторяется почти дословно в Intro / Related Work / Method / Discussion / Conclusion.
- Results §6.3 «Interpretation» по тону ближе к Discussion; дробит Results и занимает место, нужное таблице LOO.
- Conclusion почти копирует Abstract — мало новой сжатой ценности.
- OWAD angle («open-world», evolving seismic context) заявлен в Intro одним предложением и дальше не развивается (нет drift / unseen event types / deployment framing).

---

## 3. Несостыковки чисел и обозначений

- Текст: pooled \(\mathbb{E}[\tau]\approx 5.78\); Fig. 2: Student-\(t\) mean \(5.55\), skew mean \(6.01\) — среднее панелей сходится, но в тексте не сказано, что \(5.78\) — pooled mean по \(n=84\).
- «about \(+0.42\) on paired cells» при разнице медиан \(6.25-5.64=0.61\) и разнице means \(0.46\): источник \(0.42\) неясен; нужна точная метрика (mean paired \(\Delta\), median paired \(\Delta\)).
- Fig. 2 подписи осей: в PDF вместо \(\mathbb{E}[\tau]\) виден битый глиф / «квадратик» — проблема шрифта в SVG/PDF экспорте.
- Календарная формула \(9-\tau\): для median \(5.80\) получается \(3.2\) дня; текст говорит «about three days» — ок, но лучше писать явно \(9-\mathbb{E}[\tau]\).
- В Data: «profile parameters are defined in the Methods section», тогда как секция называется **Method** (без s).
- В plate diagram (Fig. 1) фигурирует \(\nu_f\), хотя skew-normal в основном анализе не имеет \(\nu\); диаграмма слегка вводит в заблуждение.
- Тикz: \(t=1{\ldots}T\), в тексте окно длины 8 и \(\mathcal{T}=\{2,\ldots,8\}\) — \(T\) vs 8 не сведено явно на рисунке.

---

## 4. Опечатки, язык, стиль EN

- `Passyuk rats` — нестандартное англоязычное имя линии; нужна проверка (возможно локальное название / транслит). Иначе рецензент отметит как error.
- `L-ACRD E 14-440` — с высокой вероятностью опечатка бренда **L-CARD** (в RU-тексте та же ошибка размножена).
- `Institute station of the Kamchatka Branch...` — неуклюжий кальк; лучше «the Institute station / observational site of the Kamchatka Branch...».
- `nan-padding export` — жаргон пайплайна в Experimental Setup; для paper лучше «NaN-masked days retained in the export without imputation».
- `bake-off among discarded specifications` — разговорный тон для IEEE.
- `visual ethogram` в Conclusion vs `visual behavioral proof` в Abstract — термины плавают.
- Em-dash: смесь `---` / Unicode `—` / hyphen; лучше унифицировать.
- Citation style author–year без скобок вокруг года иногда слипается с текстом (`...prediction Geller [1997]`); для IEEE чаще numeric `[1]`. Сейчас `natbib` + `plainnat` — ок как стиль, но не ICDM-default.

---

## 5. Форматирование и вёрстка

- Figures 3–5 **уплыли**: Fig. 3 оказывается после начала Discussion, Fig. 4 — между Conclusion и References, Fig. 5 — внутри References. Классический float disaster; нужны `[t]`/`[htbp]` дисциплина, уменьшение высоты, `\FloatBarrier`, или перевод части в appendix.
- Fig. 1 (plate) занимает почти целую страницу-колонку — дорого для 8-page workshop paper.
- Diagnostic pair-figures (params + overlay) очень тяжёлые (~0.4–0.5 MB каждый); PDF ~2.3 MB. Имеет смысл vector-only / downscale.
- Много `Underfull \hbox` (badness 10000) в логе — библиография и узкие колонки; русские записи ломают межсловные пробелы.
- В двухколоночнике inline math в Method ломает абзац («From a day … profile P=…») — в PDF видно разорванную строку вокруг формулы среднего.
- Нет `\usepackage{booktabs}`-таблиц вообще — при заявленном grid из 1548 cells выглядит пусто.
- Keywords в Abstract через `\textbf{Keywords:}` вручную — в IEEE обычно `\begin{IEEEkeywords}`.
- Нет `\IEEEpeerreviewmaketitle` / anonymized title block.
- Русский через `polyglossia` без hyphenation patterns: warning в логе (`No hyphenation patterns were loaded for russian`) → плохие переносы в Rodkin/GOST.

---

## 6. Библиография (cited + `.bib` hygiene)

### 6.1 Проблемы в фактически цитируемых записях

- `Sanford2010`: ключ говорит 2010, в записи и в PDF год **2003** — ключ врёт (для cited OK по году в тексте, но путает авторов).
- `Rodkin2011`: в EN-paper цитата рендерится как «Родкин [2011]» кириллицей; для английской submission лучше cite English translation (`Izvestiya, Physics of the Solid Earth`) как primary bibliographic form.
- `GOST2017`: в тексте «ГОСТ Р 57546–2017 [2017]» — кириллический author-field; для EN нужен английский bibliographic form / note («Russian State Standard…»).
- `Adams2007`: только arXiv; нет `archivePrefix` consistency / DOI; journal field = «arXiv preprint…» — допустимо, но стоит пометить как preprint явно или найти published version.
- `Sinkovec2022`: PhD thesis, автор «H. Šinkovec» без полного имени; URL/DOI отсутствуют.
- `Hyndman2018`: book cited for autocorrelation of daily summaries — слабо релевантно; лучше sleep/time-series dependence paper.
- Нет отдельной ссылки на PSIS paper (`Vehtari2024` есть в `.bib`, но не цитируется), хотя в тексте активно Pareto-\(\hat k\).

### 6.2 Ошибки в неиспользуемых, но «грязных» ключах `.bib` (риск при будущем cite)

- `Bradbury1998` → year `{1994}`.
- `Prendergast2002` → year `{2007}` (и авторы Wen & Prendergast).
- `Gelman2006` → year `{2007}`.
- `AlanTuringInstitute`, `ScikitLearnDevelopers`, `Hyndman2018`: `urldate = {05.06.2026}` — дата в будущем относительно привычного «сегодня» рецензента? (в мире автора 2026-08 ок, но формат DD.MM.YYYY нестандартен для BibTeX; лучше ISO).
- Куча мёртвого груза в `references.bib` (Selye, Herman, patent, scikit-learn, …) — для submission лучше slim `.bib` только с cited keys.

### 6.3 Пробелы в литературе относительно claim’ов

- Нет cite на classical site neuroseismology reports (2–4 day REM range effect).
- Нет OWAD / continual / open-world AD surveys или ключевых papers воркшопа.
- Нет современных multivariate CPD surveys beyond Aminikhanghahi2017.
- Interval-inflated Beta (IIB) не поддержан методологической ссылкой.

---

## 7. Рисунки и подписи

- Fig. 2: битый символ на оси \(x\); panel titles ок, но нет явной связи с calendar distance.
- Fig. 3: в PDF почти «orphaned caption» на отдельной странице/колонке относительно discussion text — читаемость низкая; сама фигура (если это gallery line chart) должна быть проверена на достаточный размер шрифта.
- Fig. 4–5: informative, но secondary; при лимите страниц их можно в appendix, а в main оставить 1 mean hist + 1 diagnostics.
- Captions иногда дублируют текст Results почти дословно — можно короче.
- Нет colorblind-safe check заявлен; для print IEEE лучше проверить grayscale.

---

## 8. Мелочи / nitpicks

- Abstract: «\(n=34\) after day-mask with \(K=6\)» — \(K\) ещё не определён для читателя abstract-only.
- «eight days before and eight days after» vs analysis on eight-day windows — легко прочитать как 16-day series; стоит одной фразой развести availability window vs analysis window.
- `after_reversed` в `\texttt{}` — ok, но без определения символа в Method.
- «SFU Neurotechnology Research Center» vs title-block «Research and Technology Center for Neurotechnologies» — имена центра не совпадают 1:1.
- Author list: все под одной аффилиацией; если у части авторов другая (KB GS RAS / Chebrov?), это потенциальная фактическая ошибка.
- В Keywords «neuroseismology» — нишевый термин без определения.
- Повтор `n=34` в Results первом абзаце после того, как уже сказано в Data/Setup — можно сэкономить строки под таблицу.
- `plainnat` + URL в Hoffman/Hyndman даёт длинные underfull boxes; DOI-only предпочтительнее в twocolumn.

---

## 9. Что сделать в первую очередь (приоритет правок)

1. Анонимизировать + перейти на ICDM IEEE template.
2. Починить float’ы (Figs 3–5) и битый glyph на Fig. 2.
3. Добавить 1 таблицу LOO top configurations; смягчить detection claim согласованно с failed shuffle.
4. Убрать кириллицу из in-text citations (Rodkin English form; GOST English form).
5. Починить `L-ACRD`→`L-CARD` (если подтвердится), `Methods`→`Method`, ключ `Sanford2010`→`Sanford2003`.
6. Сократить Intro↔Related Work дубли и OWAD-усилить 1 абзацем + 2–3 релевантных cite.
7. Объяснить `20/33` и dependence of windows; ethics protocol id; code/data statement.

---

## Capability-use audit

- Required skills: using-research-writing, peer-review, writing-core, verification
- Skills actually used: same (paper-orchestration не запускался — задача = review deliverable, не multi-chapter rewrite)
- Inputs consumed: `paper_en.pdf`, `paper_en.tex`, `body_en.tex`, `references.bib`, `paper_en.bbl`, `paper_en.log`, OWAD CFP page, `plan/outline.md`, `plan/project-overview.md`
- Inputs not used: full chapter `*.md` re-diff (LaTeX/PDF были source of truth)
- Artifacts produced: этот файл
- Verification: `pdftotext`/`pdfinfo`; cite-key ∩ bib; figure file presence; log underfulls; OWAD submission rules from workshop site
- Remaining risk: не открывал визуально каждую фигуру попиксельно; `Passyuk`/`L-CARD` требуют подтверждения автором
