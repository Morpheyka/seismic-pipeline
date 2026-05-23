# Bayesian model choice (Ch. 7–8) and our changepoint research

This note maps the textbook ideas from **Bayesian model choice** and **stochastic exploration of the model space** to what we implemented for **REM-profile changepoint** models in `rem_profiles_export_10days_lib.py` and `otchet_V_0.1.ipynb`.

---

## Chapter 7 — Model choice, uncertainty, and averaging

### What the chapter emphasizes

- **Many candidate models** (e.g. different predictors in regression) imply **model uncertainty**: picking a single “best” model can understate uncertainty.
- **BIC** and related criteria approximate **marginal likelihood** / penalize complexity for **large** \(n\) in **standard regression**; **Bayes factors** compare \(p(\text{data} \mid M_1)\) vs \(p(\text{data} \mid M_2)\).
- **Posterior model probabilities** \(p(M_m \mid \text{data})\) weight models; **Bayesian model averaging (BMA)** averages predictions or parameters over that posterior.

### What we did in our setting

Our “models” are **not** OLS regressions with closed-form BIC/BAS enumeration. Each model is a **discrete configuration**:

- REM profile construction: \((\text{window\_hours}, \text{step\_hours}, \text{rem\_stage})\),
- chunking: `n_chunks`,
- **feature design**: which **group × metric** blocks enter the likelihood (e.g. `concat` mean + `odd` range),
- **likelihood and priors** per metric (`parameter_selection`).

We **do not** plug our PyMC changepoint likelihood into the book’s BIC formulas. Instead we adopted the **same conceptual layer**:

1. **Explicit model list** — each configuration is a full generative model.
2. **Model uncertainty** — we explore many configurations rather than assuming one is known.
3. **Weights / summaries over visited models** — `summarize_model_search` gives visit frequencies and best scores; this is analogous in spirit to reporting **which structures the data supports**, without claiming a fully calibrated \(p(M \mid y)\) from BIC.

**Takeaway:** Ch. 7 motivates *treating the model as uncertain* and *comparing models on evidence + parsimony*; our implementation realizes that idea in a **custom changepoint** likelihood via **scoring and visitation summaries**, not via BIC on linear regression.

---

## Chapter 8 — Stochastic model space search and flexible \(g\)

### What the chapter emphasizes

- When the **model space is large**, **enumerate all models** is infeasible; **Metropolis–Hastings** (or similar) can **walk over models** with acceptance based on **posterior odds / Bayes factors** (or proxies).
- **Prior on the model** matters; **proposals** should be designed (add/drop/swap variables) and corrected for asymmetry if needed.
- **Zellner’s \(g\)-prior** and **robust choices for \(g\)** (unit information, **Zellner–Siow**, **hyper-\(g/n\)**) address sensitivity and paradoxes when \(g\) is mishandled.

### What we did in our setting

#### 1. Metropolis–Hastings over discrete models (outer loop)

- **Outer chain:** `metropolis_hastings_model_search` proposes **one local change** at a time (REM params, `n_chunks`, add/remove a feature block, change likelihood, rescale a prior hyperparameter).
- **Inner inference:** for each proposed model, **PyMC NUTS / BlackJAX** samples **continuous** parameters; with JAX backends we use **`tau_mode='marginalized'`** so \(\tau\) does not break differentiability.
- **Accept/reject** uses **`changepoint_log_target(score_changepoint_trace(...))`**, a **hand-built score** dominated by **\(P(\tau > 7)\)** plus diagnostics (e.g. \(\hat R\), ESS) and a mild complexity penalty — **not** a formal marginal likelihood ratio.

**Takeaway:** This mirrors Ch. 8’s **“MCMC over models”** workflow, with the honest caveat that our acceptance target is a **research score aligned with domain goals**, not the textbook’s exact BF for linear models.

#### 2. Flexible \(g\)-style priors on regime means (per metric)

For likelihoods with **location** \(\mu\) modeled through **Normal priors** on the two regimes (`normal`, `student_t`, `lognormal` blocks), each metric may set:

```text
g_prior: { type: none | unit_information | hyper_g_n | zellner_siow, ... }
```

Prior scale on \(\mu_{1},\mu_{2}\) is \(\sigma_0\sqrt{g}\) with \(\sigma_0\) from `mu_prior["sigma"]`, implementing the **“unknown overall prior spread controlled by \(g\)”** idea from Ch. 8 in a **scalar** way suited to our non-regression likelihood.

**Takeaway:** We imported **hyper-\(g/n\)** and **Zellner–Siow**-style *stochastic* \(g\), plus **unit-information** \(g=n\), as optional **per-feature** controls — default **`none`** preserves previous behavior.

#### 3. Mixed feature models (multi-block likelihood)

The book’s variable selection is “which predictors enter the regression.” Our analogue is **which engineered feature blocks** enter the joint model **with a single shared \(\tau\)**.

- **Implementation:** `feature_selection` as a **dict of groups → list of metrics** (subject to equal chunk counts across selected groups).
- **Presets:** `FEATURE_SELECTION_PRESETS` includes examples such as **`concat_mean_odd_range`** (`concat` mean + `odd` range).

**Takeaway:** Same **composition-of-evidence** idea as including multiple predictors, but here blocks are **different summaries of the same underlying REM trajectory**.

---

## Visualization and reporting

- **`plot_model_search_results`** plots the MH chain: log-target, running acceptance, \(P(\tau > \tau_{\text{thr}})\) and MAP \(\tau\), feature visit shares, and likelihood visit shares.
- This supports **diagnostic reading** in the spirit of Ch. 8’s convergence / inclusion plots, adapted to our **custom score** and **discrete** model path.

---

## Practical limits (what we did *not* claim)

1. **No exact BMA** over a closed-form marginal likelihood for the full changepoint model.
2. **MH target is not a proper posterior over models** unless one derives a consistent joint model across \(M\) and \(\theta\); we use an **explicit scoring** construction for **scientific prioritization** (especially \(\tau > 7\)).
3. **BIC / `bas.lm`-style** machinery from the book applies to **linear regression**; we cite it as **motivation**, not as a drop-in formula.

---

## Files touched by this line of work

| Piece | Role |
|--------|------|
| `seismic_pipeline_standalone/rem_profiles_export_10days_lib.py` | Changepoint model, `g_prior` on \(\mu\), MH search, scoring, summaries, plots |
| `otchet_V_0.1.ipynb` | 10-day workflow, optional `RUN_MODEL_SEARCH`, imports |
| `literature/bayesian_ch7_ch8_research_summary.md` | This mapping document |

---

*If you later want this text inside the notebook as a markdown cell, it can be pasted or split into shorter cells for readability.*
