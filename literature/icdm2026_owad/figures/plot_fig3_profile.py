#!/usr/bin/env python3
"""Fig. 3: mean-only E[τ] vs overlap for N=12 and N=24. No gallery titles."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "mean_only_profile_agg.csv"
OUT = HERE.parent / "latex" / "images" / "fig_mean_by_profile"

COLORS = {"student_t": "#F28E2B", "skew_normal": "#4E79A7"}
LABELS = {"student_t": r"Student-$t$", "skew_normal": "skew-normal"}


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
            "legend.fontsize": 8,
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


def main() -> None:
    setup_style()
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, n_pts in zip(axes, (12, 24)):
        sub = df[df["n_points"] == n_pts]
        for lik in ("student_t", "skew_normal"):
            g = sub[sub["mean_lik"] == lik].sort_values("overlap")
            ax.errorbar(
                g["overlap"],
                g["mean_e"],
                yerr=g["std_e"],
                color=COLORS[lik],
                marker="o",
                ms=5,
                lw=1.4,
                capsize=2.5,
                label=LABELS[lik],
            )
        ax.set_title(rf"$N={n_pts}$", loc="left", fontweight="bold")
        ax.set_xlabel("overlap")
        ax.set_xticks([0.0, 0.25, 0.50])
        ax.set_xlim(-0.06, 0.56)
        ax.set_ylim(5.0, 7.0)
    axes[0].set_ylabel(r"mean $\mathbb{E}[\tau]$")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(OUT.with_suffix(ext))
        print(f"wrote {OUT.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
