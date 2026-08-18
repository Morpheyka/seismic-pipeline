# Финальное ревью `paper_en.pdf` (ICDM 2026 OWAD)

Дата: 2026-08-14. Объект: `literature/icdm2026_owad/latex/paper_en.pdf` (сборка 14 Aug 2026 16:33, 8 стр., US Letter, IEEEtran conference).

Вердикт: **minor revision перед сабмитом**. Desk-reject по длине / IEEE 2-col / анонимному титулу с ревью 12 августа снят. Научный каркас читаемый, но фигуры ещё не «камерные», а для OWAD не хватает внешнего baseline.

Сабмитить нужно **`paper_en.pdf`**, не `paper_en_camera.pdf` (имена, 9 стр.).

---

## Общая оценка

**Исследование.** Байесовская одноточечная разладка \(\tau\) на восьмидневных event-aligned средних REM у крыс (\(n=34\)); PSIS-LOO по сетке профилей; within-window day-shuffle как контроль порядка суток; IIB только как вторичная проверка размаха.

**Рекомендация.** Не desk reject по формату. Для воркшопа — слабый, но допустимый case study по regime-onset / time-series AD, если явно не обещать continual / open-world adaptation. Перед порталом стоит почистить фигуры (1–2 часа) и одной фразой закрыть \(\hat k>1\) и «Pareto IDs».

**Ключевые плюсы.**

1. Анонимный IEEE two-column, numeric cite, 8 страниц: контент заканчивается на с. 7, ссылки на с. 7–8 (лимит 8+2 выдержан).
2. Mean-first сюжет согласован с title; non-claim «не оперативный прогноз» и слабая календарная каузальность проведены до Conclusion.
3. Есть Table I (top-10 LOO), Fig. 5 (20 seeds), синтетическая мощность null, Data/Code Availability.

**Ключевые минусы.**

1. Fig. 3 всё ещё с gallery-заголовком `A — onset vs overlap (improved baseline)`.
2. Fig. 4 и 6 несут fingerprint пайплайна в title (`4f6e4c…`, `00a650…`).
3. Fig. 5 содержит Type 3 шрифты (риск IEEE PDF eXpress на camera-ready).
4. Нет внешнего CPD/AD baseline; Pareto-\(\hat k_{\max}=1.08\) в Table I не обсуждён; обещанные Pareto-ID в Results не приведены.

---

## Соответствие OWAD / ICDM (верификация)

Источник правил: [OWAD CFP](https://sites.google.com/view/icdm-owad-2026-workshop/home), дедлайн 20 Aug 2026.

| Требование | Факт | Статус |
|---|---|---|
| IEEE 2-column | `\documentclass[conference]{IEEEtran}`, 10 pt, Letter 612×792 | pass |
| Анонимизация (triple-blind) | `Anonymous Authors`; Data без станции / SFU / протокола / координат / дат июля | pass с оговоркой |
| ≤8 стр. контента + 2 на ссылки | 8 стр. всего; References с низа с. 7 | pass |
| Desk reject: anonymity / length / format | формальные триггеры закрыты | pass |
| ICDM template | IEEEtran conference, не zip ICDM 2026; XeLaTeX + Liberation Serif вместо pdflatex+Times | приемлемо для review; camera может спросить PDF eXpress |
| Relevance to OWAD | RW цитирует Chandola/Gama/Han/Faber; явный non-claim: не continual adaptation | слабый fit, но заявлен честно |

Оговорки по анонимности.

- В титуле написано «double-blind review», у OWAD — **triple-blind**. Это формулировка, не идентичность.
- Цитата Saevskiy et al. 2025 [14] + уникальная схема «крысы + REM + сейсмика» может деанонимизировать специалистам. Для triple-blind это стандартный остаточный риск, не грубый self-cite «our previous work».
- Camera-копия (`paper_en_camera.pdf`) именная и на 9 страницах — только для внутреннего/после acceptance.

---

## Рисунки и схемы (визуальная проверка всех 8 страниц)

Сборка: float’ы больше не падают в References (это было критично 12 августа). Размещение сейчас: Fig. 1 с. 4; Fig. 2–3 с. 5; Fig. 4–5 с. 6; Fig. 6 с. 7 вместе с Discussion. Для IEEE нормально, Fig. 1 отстаёт на одну страницу от первого упоминания.

### Fig. 1 — plate (`tau_model_en.tikz`)

Читается, легенда random / deterministic / observed на месте, dashed box про маргинализацию \(\tau\) понятен. Два смысловых сдвига относительно текста: узел \(\nu_f\) есть и у skew-normal (у которого \(\nu\) нет); пластина времени подписана \(t=1{\ldots}T\), тогда как окно фиксировано длиной 8 и \(\mathcal{T}=\{2,\ldots,8\}\). Для 8-страничника схема широкая (`figure*`, 0.78 textwidth), но не ломает колонку.

### Fig. 2 — гистограммы \(\mathbb{E}[\tau]\)

Ось \(\mathbb{E}[\tau]\) визуально читается (STIX); битый «квадратик» с ревью 12 августа на печати не воспроизводится. В PDF-тексте глиф всё ещё в STIX NonUnicode — это артефакт извлечения, не обязательно дыра на бумаге. Заголовок внутри рисунка аргументативный (`does not pile up at the prior edge`); для IEEE лучше нейтральный title или только caption.

### Fig. 3 — onset vs \(N\), overlap — **править**

На самом графике остался служебный заголовок галереи: **«A — onset vs overlap (improved baseline)»**. Для submission это выглядит как черновик. PDF, судя по `pdffonts`, почти без текстового слоя (matplotlib outlines / raster) — на печати ок, для копирования подписей плохо.

### Fig. 4 — \(\nu\)-split diagnostics — **править title**

Панели \(\mu,\sigma,\nu,\tau\) и overlay читаются. В title торчит fingerprint `4f6e4c855d72864d_nu_split | daily: mean | mean=student_t;nu_per_regime`. Для рецензента это шум пайплайна; легенда overlay мелкая в одну колонку.

### Fig. 5 — within-window shuffle

Сюжет верный: real peak на \(\tau=6\)–7 vs shuffle peaks на краях 2 и 8. **Type 3 fonts** в `fig_null_within_window.pdf` (`pdffonts`: DejaVu Type 3). Для review обычно проходит; IEEE PDF eXpress на camera-ready Type 3 часто режет. Пересобрать с `pdf.fonttype=42` / STIX, как Fig. 2.

### Fig. 6 — IIB range

Диагностика согласуется с текстом (пик \(\tau=7\), \(\mathbb{E}[\tau]=6.88\), \(\pi\) выше до onset). Снова fingerprint в title (`00a650a4c85c7621 | daily: range | …`). Overlay-легенда плотная; в две колонки на с. 7 ещё читается, в grayscale «before/after» могут слиться.

### Table I

Влезает в колонку, booktabs, числа стыкуются с prose. Rank 1: daily / Student-\(t\) / \(N=24\) / ov=0 / elpd 0.480 / \(\mathbb{E}[\tau]=6.03\). Confirmatory в тексте даёт \(\mathbb{E}[\tau]\approx 5.74\) (shared \(\nu\)) — это другой бюджет MCMC, не ошибка, но одной оговорки в caption/тексте не хватает. \(\hat k_{\max}=1.08\) (rank 2) и \(1.03\) (rank 7) выше порога надёжности PSIS (\(>0.7\) тревога, \(>1\) оценка elpd по сути сломана) — в Results это не сказано.

---

## По разделам

### Abstract / title

Title больше не обещает multivariate series — совпадает с mean-first. Abstract точно отражает дизайн, null и non-claim. Для abstract-only читателя \(K=6\) ещё не определён; это nitpick.

### Introduction / Related Work

Мотивация → физиологический маркер → CPD → OWAD-мост выстроен. Дублирование Intro↔RW сохраняется (precursors, REM/stress, non-prediction). Для 8 страниц это дорого, но не ломает логику. OWAD-абзац честный: не выдаёт работу за continual AD.

Классические нейросейсмоотчёты той же станции (окно 2–4 суток, range) по-прежнему **без bib-ключа**.

### Data / Method / Setup

Анонимная ветка Data аккуратная. `L-CARD`, *Rattus norvegicus* исправлены. Жаргон `nan-padding export` в Setup остался. Method: единый \(\tau\), маргинализация, чтение края prior как «один режим» — согласовано с Results/null. IIB без методологической ссылки. Обещание «influential windows reported by ID» в Method/Setup **не исполнено** в Results.

Screening MCMC (tune 1000 / draws 500 / 2 chains) на 1548 ячеек по-прежнему лёгкий; сводки \(\hat R\)/ESS по сетке нет.

### Results

Числовая линия mean-only → LOO → confirmatory \(\nu\)-split → shuffle → IIB согласована. Detection claim умеренный: mid-window onset чувствителен к порядку суток, календарная каузальность слабая. Не хватает: (i) разведения screening \(\mathbb{E}[\tau]=6.03\) vs confirmatory 5.74; (ii) комментария к \(\hat k>1\); (iii) любого внешнего baseline (даже CUSUM/offline MAP на той же сетке \(\tau\)).

Подсекция C Interpretation по тону ближе к Discussion.

### Discussion / Conclusion

Ограничения названы (\(n=34\), одна разладка, нет биомаркеров стресса, нет прогноза). Conclusion почти пересказывает Abstract — для workshop допустимо, новой сжатой ценности мало.

---

## Статистика и воспроизводимость

- Единица LOO = event–animal pair при нескольких животных на одни даты: кластерная зависимость не моделируется.
- Null: синтетика показывает мощность; 20 seeds на реальных окнах ломают mid-window консенсус. Это сильнее, чем failed 3-seed mean-PMF из промежуточного черновика.
- Code/data: «will be released upon acceptance» — для review слабовато; ICDM просит reproducibility checklist на основном треке, у воркшопа это мягче, но anonymous repo не помешал бы.
- Cited keys \(\subseteq\) `references.bib`: 27 ключей, missing 0. `Vehtari2024` в bib есть, в тексте не цитируется, хотя Pareto-\(\hat k\) — центральный диагностический термин.

---

## Язык / writing-core (EN)

Механических `firstly/secondly/it is worth noting` нет. Повторы non-claim и IIB-обоснования всё ещё частые, но уже не дословный copy-paste на каждую секцию. `visual ethogram` (Conclusion) vs `visual behavioral proof` (Abstract) — термины плавают. Em-dash в целом единообразный (`---`).

---

## Что закрыто с ревью 12 августа

Закрыто: анонимный титул; IEEEtran; IEEEkeywords; numeric cite; LOO-таблица; float’ы не в References; Fig. 2 glyph на печати; `L-CARD`; `Sanford2003`; Rodkin/GOST English form; Data/Code; ethics id в camera; IIB 20/34; within-window null + ослабленный/уточнённый claim; OWAD cites; mean-first title.

Не закрыто или частично: baselines; classical site-report cite; Pareto IDs / \(\hat k>1\); Intro↔RW дубли; gallery-title Fig. 3; fingerprint titles; Type 3 на Fig. 5; `double-blind` wording.

---

## Приоритет правок до 20 августа

1. Снять gallery-title с Fig. 3; убрать fingerprints из Fig. 4/6; пересобрать Fig. 5 без Type 3.
2. Одна–две фразы: \(\hat k_{\max}>1\) на top-ячейках; screening vs confirmatory \(\mathbb{E}[\tau]\); либо показать Pareto-ID, либо убрать обещание «reported by ID».
3. Титул: `triple-blind review`.
4. Если останется ~0.3 стр.: крошечный baseline (offline MAP / CUSUM на той же дискретизации) — это главный научный запрос OWAD-рецензента, не блокер формата.
5. Camera: не слать в портал; PDF eXpress — после acceptance.

---

## Верификация (команды)

- `pdfinfo paper_en.pdf` → Pages=8, 612×792 pts (letter), 10 pt IEEEtran (лог: “This is a 10 point document”).
- `pdftotext` + grep имён/станции/координат → в анонимном PDF нет Ponomarev/SFU/Kamchatka/53.06; есть [14] Saevskiy в списке литературы.
- Cite ∩ bib → missing [].
- `pdffonts fig_null_within_window.pdf` → Type 3.
- Визуально все 8 страниц (`pdftoppm -r 150`) + исходники `latex/images/`.
- OWAD CFP: 8+2, IEEE 2-col, anonymized, desk reject за format/anonymity/length.
