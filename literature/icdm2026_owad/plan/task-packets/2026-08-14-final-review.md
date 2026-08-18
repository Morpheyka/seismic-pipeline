## Task Packet

- Scope: финальное ревью `latex/paper_en.pdf` (ICDM 2026 OWAD): научная состоятельность, вёрстка/рисунки, соответствие требованиям воркшопа. Не полный redraft глав.
- Files to read: `latex/paper_en.pdf`, `paper_en.tex`, `body_en.tex`, `paper_en_camera.tex`, `references.bib`, `latex/images/*`, `plan/review_paper_en.md`, OWAD CFP.
- Files allowed to edit: `plan/review_paper_en_final.md`, `plan/progress.md`, `plan/notes.md`, этот packet. Правки PDF/LaTeX — только если пользователь попросит после ревью.
- Required skills: paper-orchestration, peer-review, writing-core, latex-output, figures-python, figures-diagram, verification.
- Evidence/data inputs: OWAD CFP (8+2, IEEE 2-col, triple-blind, desk reject); предыдущее ревью 2026-08-12.
- Required artifacts: письменный review report; visual check всех страниц PDF и фигур; verification commands (pdfinfo, pdftotext, anonymization scan, log, cite∩bib).
- Rejection checks: не выдумывать данные/литературу; не заявлять «готово к сабмиту» без верификации; не править рукопись без запроса.
- Validation commands:
  - `pdfinfo paper_en.pdf` (pages, letter)
  - `pdftotext` + anonymization grep
  - page-to-image visual inspection
  - latex log overfull/undefined
  - cite keys ⊆ bib
