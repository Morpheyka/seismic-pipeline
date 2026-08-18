## Task Packet

- Scope: финальное ревью `latex/paper_en.pdf` перед подачей. Критерии: ICDM 2026 Research Track CFP (https://icdm2026.neu.edu.cn/11666/list.htm) + OWAD CFP (реальная площадка: 8+2, IEEE 2-col, triple-blind). Особый акцент: PDF print-check для Asian/European glyphs. Не полный redraft глав. Правки PDF/LaTeX — только если пользователь попросит после ревью.
- Files to read: `latex/paper_en.pdf`, `paper_en.tex`, `body_en.tex`, `paper_en_camera.tex`, `references.bib`, `paper_en.bbl`, `latex/images/*`, `plan/review_paper_en_final.md`, ICDM CFP, OWAD CFP.
- Files allowed to edit: `plan/review_paper_en_icdm_submit.md`, `plan/progress.md`, `plan/notes.md`, этот packet.
- Required skills: paper-orchestration, peer-review, writing-core, latex-output, figures-python, figures-diagram, verification.
- Evidence/data inputs: ICDM 2026 Research Track CFP; OWAD CFP; IEEE 2-column template; ревью 2026-08-14 и правки 2026-08-17.
- Required artifacts: письменный review report; visual check всех страниц PDF и фигур; verification (pdfinfo, pdftotext, pdffonts, anonymization, cite∩bib, page raster).
- Rejection checks: не выдумывать данные/литературу; не заявлять «готово к сабмиту» без верификации; не править рукопись без запроса.
- Validation commands:
  - `pdfinfo paper_en.pdf` (pages ≤10 research / 8+2 OWAD, letter)
  - `pdffonts` (no Type 3; embed; Cyrillic/European glyphs)
  - `pdftotext` + anonymization grep
  - `pdftoppm` visual inspection all pages
  - latex log overfull/undefined
  - cite keys ⊆ bib
