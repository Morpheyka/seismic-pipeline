#!/usr/bin/env python3
"""Plot Variant A: real vs within-window-shuffle mean p(τ=k).

Reads ``within_window_shuffle_results.json`` and writes SVG/PDF/PNG for
``fig:null-within-window`` (mockup: variant_A_mean_pmf_real_vs_shuffle).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STANDALONE = Path(__file__).resolve().parent.parent
REPO = STANDALONE.parent
DEFAULT_RESULTS = (
    STANDALONE
    / "run_output_8day_density_safe"
    / "within_window_shuffle"
    / "within_window_shuffle_results.json"
)
DEFAULT_OUT_STEMS = [
    REPO / "literature/icdm2026_owad/latex/images/fig_null_within_window",
    REPO / "literature/icdm2026_owad/figures/null_within_window/fig_null_within_window",
    STANDALONE
    / "run_output_8day_density_safe"
    / "within_window_shuffle"
    / "fig_null_within_window",
]

COLOR_REAL = "#1f4e79"
COLOR_SHUFFLE = "#6b7280"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS)
    p.add_argument(
        "--out-stem",
        type=Path,
        action="append",
        default=None,
        help="Output path without extension (repeatable). Defaults to latex+figures.",
    )
    return p.parse_args()


def _load_pmfs(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    real = next(r for r in rows if r.get("arm") == "real")
    agg = next((r for r in rows if r.get("arm") == "shuffle_seeds_mean"), None)
    if agg is None:
        shuf = [r for r in rows if r.get("arm") == "within_window_day_shuffle"]
        if not shuf:
            raise SystemExit("No shuffle rows in results JSON")
        support = [int(k) for k in shuf[0]["tau_support"]]
        stack = np.vstack([np.asarray(r["mean_p_tau"], dtype=float) for r in shuf])
        shuffle_pmf = stack.mean(axis=0)
    else:
        support = [int(k) for k in agg["tau_support"]]
        shuffle_pmf = np.asarray(agg["mean_p_tau"], dtype=float)
    real_pmf = np.asarray(real["mean_p_tau"], dtype=float)
    return support, real_pmf, shuffle_pmf


def plot_variant_a(
    support: list[int],
    real_pmf: np.ndarray,
    shuffle_pmf: np.ndarray,
    out_stems: list[Path],
) -> None:
    uniform = 1.0 / float(len(support))
    x = np.arange(len(support))
    width = 0.72

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=True)
    panels = [
        (axes[0], real_pmf, COLOR_REAL, "Real (event-aligned)"),
        (axes[1], shuffle_pmf, COLOR_SHUFFLE, "Within-window day-shuffle"),
    ]
    ymax = max(float(real_pmf.max()), float(shuffle_pmf.max()), uniform) * 1.18

    for ax, pmf, color, title in panels:
        ax.bar(x, pmf, width=width, color=color, edgecolor="white", linewidth=0.6, zorder=2)
        ax.axhline(uniform, color="#333333", linestyle="--", linewidth=1.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in support])
        ax.set_xlabel(r"$\tau$ (day index)")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.0, ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)

    axes[0].set_ylabel(r"mean $p(\tau=k)$")
    fig.suptitle(
        r"Null control: mean $p(\tau=k)$ (dashed = uniform $1/|\mathcal{T}|$)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    for stem in out_stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".svg", ".pdf", ".png"):
            out = stem.with_suffix(ext)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[plot] wrote {out}")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    stems = args.out_stem if args.out_stem else DEFAULT_OUT_STEMS
    support, real_pmf, shuffle_pmf = _load_pmfs(args.results_json)
    plot_variant_a(support, real_pmf, shuffle_pmf, stems)
    mad_r = float(np.mean(np.abs(real_pmf - 1.0 / len(support))))
    mad_s = float(np.mean(np.abs(shuffle_pmf - 1.0 / len(support))))
    print(
        f"[plot] MAD_uniform real={mad_r:.4f} shuffle={mad_s:.4f} "
        f"(shuffle closer to uniform if MAD_s < MAD_r)"
    )


if __name__ == "__main__":
    main()
