# IIB range regime summary (confirmatory refit, density-safe n=34)

Source: `seismic_pipeline_standalone/run_output_8day_density_safe/refit_iib_range_n34/rank18_00a650a4c85c7621/observations.npz`  
Fingerprint: `00a650a4c85c7621` (`daily:range`, IIB@0.9, \(N=12\), \(\mathrm{ov}=0\))  
Protocol: density-safe nan-pad + day-mask \(K=6\) → \(n=34\) (legacy confirmatory export with `drop_incomplete_events=True` had wrongly dropped `R3 / 2025-01-23 / after_reversed`, yielding \(n=33\)).  
Scale: \(y = \mathrm{range}/2\) (unit interval; matches likelihood support)  
Split: \(\tau_{\mathrm{MAP}}=7\) → before = columns \(t<7\) (6 days), after = columns \(t\ge 7\) (2 days)  
Posterior: \(\mathbb{E}[\tau]\approx 6.88\), MAP concentration \(\approx 0.56\) (tune 6000 / draws 3000 / chains 4)

| Regime | \(n\) (day-obs) | median | mean | share \(y\ge 0.9\) | share \(y\ge 0.95\) | share \(y=1\) |
|--------|----------------:|-------:|-----:|-------------------:|--------------------:|--------------:|
| before | 196 | 0.753 | 0.752 | 21.4% | 11.7% | 6.1% |
| after | 68 | 0.711 | 0.690 | 11.8% | 2.9% | 1.5% |

Event-wise (\(n=34\) windows): median(after) \(<\) median(before) in **20 / 34** (58.8%).

Notes for prose:
- Ceiling mass near 1 is concentrated in the pre-\(\tau\) regime; post-\(\tau\) days sit lower on the unit interval.
- July multi-animal dates (R1–R4) are fully present in the \(n=34\) cohort.
