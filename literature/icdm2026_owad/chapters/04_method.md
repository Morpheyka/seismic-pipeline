# 4. Method

Daily REM profiles are built from the hypnogram as the fraction of time spent in paradoxical sleep in successive temporal bins. A profile is parameterized by the number of bins per day \(N\) and the overlap fraction \(\mathrm{ov}\) of adjacent windows; the twelve non-overlapping two-hour bins correspond to \((N,\mathrm{ov})=(12,0)\). From a day profile \(P=(p_1,\ldots,p_N)\) we compute the mean \(\bar P=N^{-1}\sum_i p_i\) and, when needed for secondary checks, the range \(R=\max_i p_i-\min_i p_i\), including half-day (light/dark) versions of the same summaries. Mean and range are two separate scalar features, not a sum: when both are active they share one common \(\tau\). The primary analysis uses mean features, which after window-wise scaling to \([-1,1]\) yield real-valued series with stable onset posteriors. Artifact days and days without a hypnogram are masked as NaN and skipped in the likelihood without imputation and without shrinking the discrete grid of \(\tau\). A window enters the sample if at least \(K=6\) days remain valid.

For a feature vector \(\mathbf{x}_t\) on days \(t=1,\ldots,8\) we use a single regime switch

\[
\mathbf{x}_t \sim
\begin{cases}
p(\mathbf{x}_t\mid\boldsymbol{\theta}_1), & t < \tau,\\
p(\mathbf{x}_t\mid\boldsymbol{\theta}_2), & t \ge \tau.
\end{cases}
\]

The discrete day index is event-aligned so that \(t=8\) is the day nearest the shock (one day from the event) and \(t=1\) is eight days from the event; calendar distance in days is \(9-t\). The discrete onset \(\tau\) is given a uniform prior on \(\mathcal{T}=\{2,\ldots,8\}\) and is marginalized for NUTS sampling [HoffmanGelman2014],

\[
p(\mathbf{y}\mid\boldsymbol{\theta})
=
\frac{1}{|\mathcal{T}|}
\sum_{k\in\mathcal{T}}
p(\mathbf{y}\mid\tau{=}k,\boldsymbol{\theta}).
\]

The posterior \(p(\tau{=}k\mid\mathbf{y})\) and \(\mathbb{E}[\tau\mid\mathbf{y}]\) are recovered from posterior draws of \(\boldsymbol{\theta}\). When several features are active in one configuration, they share a common \(\tau\). Figure~\ref{fig:tau-model} shows the probabilistic plate diagram: a discrete uniform prior on \(\tau\), a deterministic regime indicator \(z_t=\mathbb{1}[t<\tau]\), feature-wise likelihoods, and analytic marginalization of \(\tau\) before NUTS. This construction tests for a seismic-event-associated anomaly as a changepoint in short brain-derived REM summaries—an instrumented physiological signal rather than a visual behavioral report.

Likelihoods match the support of each feature. For mean features we use Student-\(t\), which accommodates heavier tails and a few outlying days, and skew-normal, which allows asymmetry in the day-to-day distribution of means. On the discrete grid \(\tau\in\{2,\ldots,8\}\), in the absence of a regime-shift anomaly one expects either a near-uniform posterior on \(\tau\) or edge peaks at \(\tau=2\) or \(\tau=8\): with little evidence for a two-regime split the model prefers to place all days in one shared-parameter distribution. Mid-window mass is therefore the readable signature of an onset. Both mean likelihoods keep \(\mathbb{E}[\tau]\) in that mid-window band on mean-only series. With few bins and min–max normalization over the eight-day window, daily range often concentrates near one, where an ordinary Beta density diverges at the upper boundary; secondary range checks therefore use an interval-inflated Beta (IIB@\(0.9\)) rather than a plain Beta on \((0,1)\).

Generalization among configurations is scored by PSIS-LOO [Vehtari2017, Vehtari2024], with the observation unit equal to an event–animal pair and a normalized score \(\overline{\mathrm{elpd}}/(F\cdot E')\) inside a fixed observation builder (day mask on, \(K=6\)). LOO ranks profile geometry and mean likelihood choices. Windows with large Pareto-\(\hat k\) are retained rather than dropped; for cells with \(\hat k_{\max}>1\), the elpd values are treated only as an ordinal ranking among mean specifications [Vehtari2024]. Sampling uses NUTS under the marginalized \(\tau\) model; we monitor \(\hat R\), effective sample size, and Pareto-\(\hat k\). The combinatorial grid over profile geometry is specified next.
