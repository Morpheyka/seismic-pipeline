# Blueprint — Experimental Setup (OWAD)

### P1 Goal of design
- Exhaustive screen of preprocessing × likelihoods → τ-diagnosis screen on range families → stability inside IIB

### P2 Exploratory grid
- N∈{12,24}, ov∈{0,0.25,0.5}; daily/day/night × mean/range (≤3 blocks); mean lik; range lik; τ∈{2..8}; mask ON K=6; 1548 ok; 1464 with range / 366 per family

### P3 Screening rule + analysis set
- E[τ] not piled at τ=2; no boundary-driven elpd inflation; IIB analysis set; others as controls; mean-only contrast

### P4 Confirmatory / sensitivity / neg-control
- Primary A/B; mask OFF; before_only; shuffle seeds 0,1,2 on A

### P5 Metrics
- E[τ], elpd/(F E'), Rhat/ESS/Pareto-k; compare elpd within one builder; no unique winner rhetoric
