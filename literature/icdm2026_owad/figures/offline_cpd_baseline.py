#!/usr/bin/env python3
"""Offline Gaussian two-mean scan on confirmatory daily:mean windows.

No MCMC. Writes a JSON summary for the Results paragraph.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OBS = (
    REPO
    / "seismic_pipeline_standalone"
    / "run_output_8day_density_safe"
    / "refit_best_mean_only"
    / "rank11_4f6e4c855d72864d"
    / "observations.npz"
)
META = OBS.parent / "refit_meta.json"
OUT = Path(__file__).resolve().parent / "data" / "offline_cpd_baseline.json"
SUPPORT = np.arange(2, 9)


def gaussian_profile_ll(y: np.ndarray, k: int) -> float:
    """Shared-variance two-mean Gaussian log-likelihood; NaNs skipped."""
    t = np.arange(1, y.size + 1)
    before = y[(t < k) & np.isfinite(y)]
    after = y[(t >= k) & np.isfinite(y)]
    if before.size < 1 or after.size < 1:
        return -np.inf
    n = before.size + after.size
    mu1 = float(before.mean())
    mu2 = float(after.mean())
    sse = float(((before - mu1) ** 2).sum() + ((after - mu2) ** 2).sum())
    sigma2 = sse / n
    if sigma2 <= 0.0:
        return -np.inf
    return -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)


def scan_window(y: np.ndarray) -> int:
    lls = np.array([gaussian_profile_ll(y, int(k)) for k in SUPPORT], dtype=float)
    return int(SUPPORT[int(np.argmax(lls))])


def main() -> None:
    if not OBS.is_file():
        raise SystemExit(f"missing observations: {OBS}")
    y = np.load(OBS)["daily__mean"]
    n, t = y.shape
    if t != 8:
        raise SystemExit(f"expected 8-day windows, got shape {y.shape}")
    tau_ml = np.array([scan_window(y[i]) for i in range(n)], dtype=int)
    meta = json.loads(META.read_text(encoding="utf-8"))
    bayes_map = int(meta.get("tau_map", 7))
    mid = np.isin(tau_ml, (6, 7))
    summary = {
        "n_windows": int(n),
        "support": SUPPORT.tolist(),
        "tau_ml": tau_ml.tolist(),
        "median_tau_ml": float(np.median(tau_ml)),
        "mean_tau_ml": float(np.mean(tau_ml)),
        "frac_mid_window_6_7": float(mid.mean()),
        "n_mid_window_6_7": int(mid.sum()),
        "bayes_map": bayes_map,
        "frac_equal_bayes_map": float(np.mean(tau_ml == bayes_map)),
        "n_equal_bayes_map": int(np.sum(tau_ml == bayes_map)),
        "counts": {str(int(k)): int(np.sum(tau_ml == k)) for k in SUPPORT},
        "note": (
            "Confirmatory daily:mean Student-t artefacts use n=33 "
            "(incomplete-day drop in that refit); paper screening cohort is n=34."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
