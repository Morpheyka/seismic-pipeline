# Blueprint — Method (OWAD)

### P1 Profiles → features
- Input: hypnogram → REM profile (N, ov) → daily/half-day mean and range → scale on valid days; day mask to NaN
- Forbidden: full artifact map dump; grid of all configs

### P2 Changepoint model
- Single switch τ on T={2..8}, shared across active features; marginalization; NUTS
- Cite: HoffmanGelman2014, carlin1992hierarchical optional

### P3 Why likelihood families
- Mean: real-valued after scaling to [-1,1]; Student-t (tails/robustness) and skew-normal (asymmetry)
- Range: compressed to (0,1); Beta support matches; plain / constrained / ZOIB / IIB probe different boundary behaviors
- IIB@0.9 rationale (author): small bin width + min–max over full 8-day window → range often near 1 → need mass near upper edge
- Screening: τ claims only for families without E[τ] piled at prior floor τ=2; IIB becomes analysis set (results deferred)

### P4 LOO + MCMC
- PSIS-LOO per event–animal; elpd normalization note; Rhat/ESS/Pareto-k; do not drop influential points
- Cite: Vehtari2017
