#!/usr/bin/env python3
"""Fig. 2: mean-only E[τ] histograms. Neutral title (no argumentative suptitle)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "mean_only_long.csv"
OUT = HERE.parent / "latex" / "images" / "fig_mean_only_by_lik"

COLORS = {"student_t": "#F28E2B", "skew_normal": "#4E79A7", "all": "#59A14F"}
LABELS = {"student_t": r"Student-$t$", "skew_normal": "skew-normal"}
BINS = np.linspace(2.0, 8.0, 25)
TAU_EDGE = 2.05
XLIM = (1.75, 8.15)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans", "Arial"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def shade_prior_edge(ax: plt.Axes) -> None:
    ax.axvspan(2.0, TAU_EDGE, color="#CC3311", alpha=0.12, zorder=0, lw=0)
    ax.axvline(2.0, color="#555555", lw=1.0, ls="--", zorder=3)


def annotate_box(ax: plt.Axes, x: np.ndarray) -> None:
    med = float(np.median(x))
    edge = float((x <= TAU_EDGE).mean())
    ax.text(
        0.98,
        0.95,
        rf"$n$ = {len(x)}" + "\n"
        + rf"median = {med:.2f}" + "\n"
        + rf"edge share = {100 * edge:.0f}\%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#dddddd",
            alpha=0.92,
        ),
    )
    ax.axvline(med, color="#222222", lw=1.1, alpha=0.8, zorder=3)


def main() -> None:
    setup_style()
    df = pd.read_csv(CSV)
    if len(df) != 84:
        raise SystemExit(f"expected 84 mean-only rows, got {len(df)}")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)

    ax = axes[0]
    x_all = df["e_tau"].to_numpy(dtype=float)
    ax.hist(
        x_all,
        bins=BINS,
        density=True,
        color=COLORS["all"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
    )
    shade_prior_edge(ax)
    annotate_box(ax, x_all)
    ax.set_title("(a)  mean-only, all families", loc="left", fontweight="bold")
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    ax.set_ylabel("density")
    ax.set_xticks([2, 4, 6, 8])

    ax = axes[1]
    for lik in ("student_t", "skew_normal"):
        x = df.loc[df["mean_lik"] == lik, "e_tau"].to_numpy(dtype=float)
        ax.hist(
            x,
            bins=BINS,
            density=True,
            histtype="stepfilled",
            color=COLORS[lik],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            label=rf"{LABELS[lik]} ($n$={len(x)}, med.={np.median(x):.2f})",
        )
    shade_prior_edge(ax)
    ax.set_title("(b)  mean-only by likelihood", loc="left", fontweight="bold")
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    ax.legend(loc="upper left")
    ax.set_xticks([2, 4, 6, 8])

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(OUT.with_suffix(ext))
        print(f"wrote {OUT.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
