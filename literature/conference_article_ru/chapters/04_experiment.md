# 4. Эксперимент (полный перебор)

Цель — прогнать сетку предобработки и правдоподобий на одних и тех же восьмидневных окнах, **просеять** семейства размаха по диагностике \(\tau\), затем оценить устойчивость разладки внутри прошедшего скрининг класса (IIB).

## Сетка конфигураций (exploratory)

- Профиль: \(N\in\{12,24\}\), \(\mathrm{ov}\in\{0,0{,}25,0{,}5\}\) (6 ячеек).
- Признаки: `daily` / `day` / `night` × `mean` / `range`, не более трёх блоков.
- Mean: `student_t`, `skew_normal`. Range (скрининг): plain `beta`, `beta_constrained`, IIB@0.9, ZOIB.
- \(\tau\in\{2,\ldots,8\}\), marginalized NUTS; Pareto-retry отключён.
- Day-mask ON, \(K=6\); экспорт с nanpad (`drop_incomplete_events=False`).
- Бюджет exploratory MCMC: tune \(1000\), draws \(500\), 2 цепи (скрининг); удлинённые refit — для primary-ячеек.

Завершено \(1548\) конфигураций (`status=ok`), из них с активным размахом \(1464\) (по \(366\) на каждое range-семейство). Артефакт: `run_output_8day_density_safe/exhaustive_search_parallel.csv`.

## Анализ после скрининга

- **Основной набор:** все конфигурации с `range=interval_inflated_beta` (\(n=366\)), в том числе range-only и mean+range.
- **Контроль спецификации:** гистограммы \(\mathbb{E}[\tau]\) по plain Beta / BC / ZOIB (пик у \(\tau=2\)); mean-only без range (\(n=84\)) — контраст, что средний признак сам по себе не даёт граничного артефакта.
- **Primary (для чувствительности маски):** A — daily mean+range, student_t + beta_constrained, \(N=24\), ov \(0{,}5\) (density-safe без инфляции; сравнить с IIB-скринингом); B — daily:range на \((24,0)/(24,0{,}25)/(24,0{,}5)\) с IIB (и BC/ZOIB как DIAG). Mask OFF и before_only — на согласованной когорте \(n=34\).
- Негативный контроль: proxy shuffle, семена \(0,1,2\) на Primary A.

Все сравнения elpd — внутри одного builder и одного прогона.
