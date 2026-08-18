# Task packet: убрать повторы «classical»

- Scope: targeted prose cut в EN/RU body + зеркала chapters. Не менять модель, таблицы, claim аномалии.
- Keep: один контраст в Intro (range / 12 бинов / 2–4 суток vs onset \(\tau\)); в Method определение \((N,\mathrm{ov})=(12,0)\) без слова classical.
- Drop: дубль в Related Work; Results про classical \((12,0)\) vs \(N=24\); Discussion «не классическое окно 2–4 дня».
- Verify: grep `classical`/`классическ` в `latex/body_*.tex` и `chapters/`; rebuild `paper_en.pdf`.
