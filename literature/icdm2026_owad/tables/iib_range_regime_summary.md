# IIB range regime summary (confirmatory refit)

Source: `seismic_pipeline_standalone/run_output_8day_density_safe/refit_iib_range/rank18_00a650a4c85c7621/observations.npz`  
Fingerprint: `00a650a4c85c7621` (`daily:range`, IIB@0.9, \(N=12\), \(\mathrm{ov}=0\))  
Scale: \(y = \mathrm{range}/2\) (unit interval; matches likelihood support)  
Split: \(\tau_{\mathrm{MAP}}=7\) → before = columns \(t<6\) (6 days), after = columns \(t\ge 6\) (2 days)

| Regime | \(n\) (day-obs) | median | mean | share \(y\ge 0.9\) | share \(y\ge 0.95\) | share \(y=1\) |
|--------|----------------:|-------:|-----:|-------------------:|--------------------:|--------------:|
| before | 198 | 0.736 | 0.742 | 20.7% | 10.6% | 5.6% |
| after | 66 | 0.699 | 0.682 | 12.1% | 3.0% | 1.5% |

Event-wise (\(n=33\) windows): median(after) \(<\) median(before) in **20 / 33** (60.6%).

Notes for prose:
- Ceiling mass near 1 is concentrated in the pre-\(\tau\) regime; post-\(\tau\) days sit lower on the unit interval.
- Numbers above are from the computed summary only (no invented rounding beyond one decimal place for percentages / three decimals for medians–means).
