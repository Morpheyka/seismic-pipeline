## Task Packet

- Scope: pre-submit правки по ревью 2026-08-17 (не redraft, не новый baseline BOCPD/PELT). Цель: нейтральный Fig. 2, plate без \(\nu_f\) / \(t=1{\ldots}8\), `nan-padding`, Data/Code + anonymous supplement, hyphenation, GOST capital R. Сохранить 8 стр. Letter.
- Files to read: `body_en.tex`, `body_ru.tex`, `paper_en.tex`, `paper_en_camera.tex`, tikz plates, `mean_only_long.csv`, `plot_fig3_profile.py`, `references.bib`.
- Files allowed to edit: latex EN/RU + camera; tikz; new `figures/plot_fig2_mean_hist.py`; `latex/images/fig_mean_only_by_lik.*`; chapters EN+RU setup; `references.bib`; `supplement/README.md`; `figures/data-manifest.md`; `plan/progress.md`, `plan/notes.md`.
- Required skills: paper-orchestration, writing-core, figures-python, latex-output, verification.
- Evidence/data inputs: `figures/data/mean_only_long.csv` (real, n=84); ревью `plan/review_paper_en_icdm_submit.md`.
- Required artifacts: rebuilt `paper_en.pdf` (8 стр.); нейтральный Fig. 2; anonymous supplement README.
- Rejection checks: не выдумывать cite на 2–4 day site reports; не менять научные числа; не слать camera в портал; страница ≤8.
- Validation commands: plot script; `PAPER_CAMERA=1 ./build.sh`; `pdfinfo` Pages=8; `pdffonts` no Type 3; grep pile-up / nan-padding / identity.
