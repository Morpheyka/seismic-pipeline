# 4. Эксперимент (полный перебор)

Цель экспериментального блока — сравнить density-safe конфигурации предобработки и правдоподобий на одних и тех же восьмидневных окнах и оценить устойчивость положения разладки \(\tau\). Ранжирование — по нормированному PSIS-LOO elpd внутри одного builder с day-mask ON.

## Сетка конфигураций

- Профиль: \(N\in\{12,24\}\), \(\mathrm{ov}\in\{0,0{,}25,0{,}5\}\) (6 ячеек).
- Признаки: `daily` / `day` / `night` × `mean` / `range`, не более трёх блоков.
- Mean: `student_t`, `skew_normal`. Range: `beta_constrained`, IIB@0.9, ZOIB. Plain `beta` — только DIAG.
- \(\tau\in\{2,\ldots,8\}\), marginalized NUTS; Pareto-retry отключён (окна не дропают).
- Экспорт: `drop_incomplete_events=False` (nanpad); eligibility по \(K=6\).

Оценка объёма exploratory: порядка \(500\)–\(900\) конфигураций. Primary stratum: `full` + mask ON. Чувствительность: (1) mask OFF на той же когорте окон; (2) `before_only` + mask ON.

## Primary A/B и контроли

- **Primary A:** `daily:mean+range`, student_t + beta_constrained, \(N=24\), \(\mathrm{ov}=0{,}5\).
- **Primary B:** `daily:range` на \((24,0)/(24,0{,}25)/(24,0{,}5)\) с beta_constrained и ZOIB.
- Негативный контроль: перемешивание дат / proxy row-shuffle within rat, семена \(0,1,2\) на Primary A при том же mask.
- DIAG: plain beta vs constrained / IIB / ZOIB на согласованной ячейке.

Все сравнения elpd проводят **внутри одного** observation builder и одного прогона. Артефакты: `run_output_8day_density_safe/`, `run_output_density_safe_confirmatory/`.
