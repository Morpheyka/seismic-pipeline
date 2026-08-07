# 5. Результаты

Primary-когорта после day-mask (\(K=6\)): **34** окна (из 36 запрошенных; включён R3 2025-01-23 after_reversed). Confirmatory: mask ON, \(\tau\in\{2,\ldots,8\}\).

> Полный exploratory (`run_output_8day_density_safe/`) ещё выполняется; visit rates T3 — после CSV. July plain Beta / \(N=48\) не используется как publishable claim.

## Primary A/B и чувствительность (mask ON/OFF, before_only)

| Конфиг | \(\mathbb{E}[\tau]\) | elpd/\(F E'\) |
|--------|----------------------|---------------|
| A mask ON (daily mean+range, st+BC, \(N=24\), ov \(0{,}5\)) | \(4{,}25\) | \(3{,}21\) |
| A mask OFF (та же когорта) | \(3{,}16\) | \(2{,}96\) |
| A before\_only | \(2{,}11\) | \(3{,}66\) |
| B BC \(N=24\) ov \(0\) / \(0{,}25\) / \(0{,}5\) | \(2{,}17\) / \(5{,}91\) / \(3{,}80\) | \(4{,}58\) / \(4{,}10\) / \(4{,}21\) |
| B ZOIB \(N=24\) ov \(0\) / \(0{,}25\) / \(0{,}5\) | \(2{,}00\) / \(4{,}49\) / \(4{,}44\) | \(7{,}79\) / \(5{,}90\) / \(5{,}07\) |
| DIAG plain beta (\(N=24\), ov \(0{,}5\)) | \(4{,}98\) | \(4{,}55\) |

Primary A при mask ON даёт \(\mathbb{E}[\tau]\) ближе к середине окна, чем July-полоса \(\approx 6{,}5\)–\(6{,}7\). Mask OFF и before\_only сдвигают оценку; before\_only упирается в нижнюю границу prior. Высокий elpd ZOIB (особенно ov \(0\)) и DIAG plain Beta выше density-safe BC на той же ячейке — caution у границ носителя.

## Негативный контроль (Primary A, семена 0–2)

Proxy row-shuffle within rat (умеренный MCMC):

| Arm | \(\mathbb{E}[\tau]\) | elpd/\(F E'\) |
|-----|----------------------|---------------|
| real dates | \(4{,}16\) | \(3{,}21\) |
| seed 0 | \(3{,}93\) | \(3{,}22\) |
| seed 1 | \(3{,}94\) | \(3{,}20\) |
| seed 2 | \(4{,}26\) | \(3{,}22\) |

Null на текущем бюджете **не** даёт резкого контраста. Это ограничивает силу causal claim; в тексте нет утверждения «стресс доказан».

## DIAG, Pareto, exploratory

Plain Beta вне eligible. Smoke (student_t+BC, skew_normal+ZOIB, IIB) зелёный. Pareto-retry отключён; ID влиятельных окон — в confirmatory JSON.
