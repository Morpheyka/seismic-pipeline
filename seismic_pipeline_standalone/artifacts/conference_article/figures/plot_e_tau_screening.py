"""
Скрининг семейств размаха: распределение E[τ].
Глава: Results (conference_article_ru).
Источник: artifacts/conference_article/exhaustive_search_parallel.csv (real).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
ART_DIR = FIG_DIR.parent  # artifacts/conference_article
CSV = ART_DIR / "exhaustive_search_parallel.csv"
OUT_DIR = FIG_DIR
DATA_DIR = FIG_DIR / "data"

# Cell / colorblind-safe
COLORS = {
    "interval_inflated_beta": "#4E79A7",
    "beta_constrained": "#F28E2B",
    "zero_inflated_beta": "#E15759",
    "beta": "#76B7B2",
}
LABELS = {
    "interval_inflated_beta": "IIB@0.9",
    "beta_constrained": "beta_constrained",
    "zero_inflated_beta": "ZOIB",
    "beta": "plain Beta",
}
# Порядок: сначала проблемные, IIB последним (выбранный)
ORDER = [
    "beta_constrained",
    "beta",
    "zero_inflated_beta",
    "interval_inflated_beta",
]
PANEL = ["(a)", "(b)", "(c)", "(d)"]
BINS = np.linspace(2.0, 8.0, 25)
TAU_EDGE = 2.05
XLIM = (1.75, 8.15)


def shade_prior_edge(ax: plt.Axes) -> None:
    ax.axvspan(2.0, TAU_EDGE, color="#CC3311", alpha=0.12, zorder=0, lw=0)
    ax.axvline(2.0, color="#555555", lw=1.0, ls="--", zorder=3)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans", "Lato", "Arial"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def parse_range_lik(s: str) -> str | None:
    m = re.search(r"range=([a-z0-9_]+)", str(s))
    return m.group(1) if m else None


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    feat = df["features"].astype(str)
    df["has_range"] = feat.str.contains(r"\brange\b")
    df["has_mean"] = feat.str.contains(r"\bmean\b")
    df["range_lik"] = df["likelihoods"].map(parse_range_lik)
    df["e_tau"] = pd.to_numeric(df["e_tau"], errors="coerce")
    return df.dropna(subset=["e_tau"])


def edge_frac(s: pd.Series) -> float:
    return float((s <= TAU_EDGE).mean())


def _ensure_dir(path: Path) -> None:
    if path.is_symlink() or path.is_dir():
        return
    path.mkdir(parents=True, exist_ok=True)

def save_fig(fig: plt.Figure, stem: str) -> None:
    _ensure_dir(OUT_DIR)
    _ensure_dir(DATA_DIR)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450)
    fig.savefig(OUT_DIR / f"{stem}.svg")
    print(f"wrote {stem} → {OUT_DIR}")


def fig_screening_hist(df: pd.DataFrame) -> None:
    """2×2 density histograms, shared axes, prior edge, %≤2.05."""
    sub = df[df["has_range"] & df["range_lik"].isin(ORDER)].copy()
    # shared density ymax
    ymax = 0.0
    dens = {}
    for lik in ORDER:
        x = sub.loc[sub["range_lik"] == lik, "e_tau"].to_numpy()
        h, _ = np.histogram(x, bins=BINS, density=True)
        dens[lik] = x
        ymax = max(ymax, float(h.max()) if len(h) else 0.0)
    ymax *= 1.12

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True)
    for ax, lik, letter in zip(axes.ravel(), ORDER, PANEL):
        x = dens[lik]
        color = COLORS[lik]
        ax.hist(
            x,
            bins=BINS,
            density=True,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
        )
        shade_prior_edge(ax)
        med = float(np.median(x))
        ax.axvline(med, color="#222222", lw=1.1, ls="-", zorder=3, alpha=0.8)
        frac = edge_frac(pd.Series(x))
        ax.text(
            0.98,
            0.95,
            f"$n$ = {len(x)}\n"
            f"медиана = {med:.2f}\n"
            f"доля у края = {100 * frac:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.92),
        )
        ax.set_title(f"{letter}  {LABELS[lik]}", loc="left", fontweight="bold")
        ax.set_xlim(*XLIM)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks([2, 3, 4, 5, 6, 7, 8])

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    for ax in axes[:, 0]:
        ax.set_ylabel("плотность")

    # общая подпись границы prior
    fig.text(
        0.5,
        0.01,
        r"заливка — зона $\mathbb{E}[\tau]\leq 2.05$; пунктир — $\tau{=}2$; "
        "сплошная — медиана",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.suptitle(
        r"Скрининг семейств размаха: распределение $\mathbb{E}[\tau]$"
        "\n(конфигурации с активным range, day-mask ON)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_fig(fig, "fig_screening_e_tau_hist")
    plt.close(fig)

    # data export
    rows = []
    for lik in ORDER:
        x = dens[lik]
        rows.append(
            {
                "range_lik": lik,
                "label": LABELS[lik],
                "n": len(x),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "frac_le_2_05": edge_frac(pd.Series(x)),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "fig_screening_e_tau_summary.csv", index=False)


def fig_screening_ecdf(df: pd.DataFrame) -> None:
    """Compact ECDF overlay — один кадр для статьи."""
    sub = df[df["has_range"] & df["range_lik"].isin(ORDER)].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xs = np.linspace(2.0, 8.0, 400)
    for lik in ORDER:
        x = np.sort(sub.loc[sub["range_lik"] == lik, "e_tau"].to_numpy())
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(
            x,
            y,
            where="post",
            color=COLORS[lik],
            lw=1.6,
            label=f"{LABELS[lik]} ($n$={len(x)})",
        )
    ax.axvspan(2.0, TAU_EDGE, color="#CC3311", alpha=0.12, zorder=0, lw=0)
    ax.axvline(2.0, color="#555555", lw=1.0, ls="--")
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    ax.set_ylabel("ECDF")
    ax.set_title(r"ECDF $\mathbb{E}[\tau]$ по семействам размаха", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.6)
    fig.tight_layout()
    save_fig(fig, "fig_screening_e_tau_ecdf")
    plt.close(fig)


def fig_iib_main(df: pd.DataFrame) -> None:
    """Основной анализ: IIB-срезы + mean-only контраст."""
    iib = df[df["range_lik"] == "interval_inflated_beta"].copy()
    mean_only = df[~df["has_range"] & df["has_mean"]].copy()
    panels = [
        ("IIB, все", iib["e_tau"].to_numpy(), COLORS["interval_inflated_beta"]),
        (
            "IIB, mean+range",
            iib.loc[iib["has_mean"] & iib["has_range"], "e_tau"].to_numpy(),
            "#59A14F",
        ),
        (
            "IIB, range-only",
            iib.loc[~iib["has_mean"] & iib["has_range"], "e_tau"].to_numpy(),
            "#B07AA1",
        ),
        ("mean-only\n(без range)", mean_only["e_tau"].to_numpy(), "#9C755F"),
    ]

    ymax = 0.0
    for _, x, _ in panels:
        h, _ = np.histogram(x, bins=BINS, density=True)
        ymax = max(ymax, float(h.max()) if len(h) else 0.0)
    ymax *= 1.12

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
    for ax, (title, x, color), letter in zip(axes, panels, PANEL):
        ax.hist(
            x,
            bins=BINS,
            density=True,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
        )
        shade_prior_edge(ax)
        med = float(np.median(x))
        ax.axvline(med, color="#222222", lw=1.1, alpha=0.8)
        frac = edge_frac(pd.Series(x))
        ax.text(
            0.97,
            0.95,
            f"$n$ = {len(x)}\nмедиана = {med:.2f}\n"
            f"доля у края = {100 * frac:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", alpha=0.92),
        )
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=8, fontweight="bold")
        ax.set_xlim(*XLIM)
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel(r"$\mathbb{E}[\tau]$")
        ax.set_xticks([2, 4, 6, 8])
    axes[0].set_ylabel("плотность")
    fig.suptitle(
        r"Основной анализ: IIB и контраст mean-only",
        fontsize=10,
        y=1.05,
    )
    fig.tight_layout()
    save_fig(fig, "fig_iib_main_e_tau_hist")
    plt.close(fig)

    rows = []
    for title, x, _ in panels:
        rows.append(
            {
                "slice": title.replace("\n", " "),
                "n": len(x),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "frac_le_2_05": edge_frac(pd.Series(x)),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "fig_iib_main_e_tau_summary.csv", index=False)


def export_long_csv(df: pd.DataFrame) -> None:
    """Длинный CSV для воспроизводимости (только нужные колонки)."""
    sub = df[df["has_range"] & df["range_lik"].isin(ORDER)][
        ["features", "likelihoods", "n_points", "overlap", "range_lik", "e_tau", "has_mean", "has_range"]
    ].copy()
    sub.to_csv(DATA_DIR / "fig_screening_e_tau_long.csv", index=False)
    mean_only = df[~df["has_range"] & df["has_mean"]][
        ["features", "likelihoods", "n_points", "overlap", "e_tau"]
    ]
    mean_only.to_csv(DATA_DIR / "fig_mean_only_e_tau_long.csv", index=False)


def main() -> None:
    setup_style()
    df = load_frame()
    export_long_csv(df)
    fig_screening_hist(df)
    fig_screening_ecdf(df)
    fig_iib_main(df)
    # manifest
    man = DATA_DIR.parent / "data-manifest.md"
    man.write_text(
        "# Data manifest (figures)\n\n"
        "Level A plots live under `seismic_pipeline_standalone/artifacts/conference_article/figures/`.\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "|---|---|---|---|---|---|\n"
        "| Screening E[τ] hist | `data/fig_screening_e_tau_summary.csv`, `fig_screening_e_tau_long.csv` | real | `../exhaustive_search_parallel.csv` | `plot_e_tau_screening.py` | `fig_screening_e_tau_hist.{png,svg}` |\n"
        "| Screening E[τ] ECDF | same | real | same | same | `fig_screening_e_tau_ecdf.{png,svg}` |\n"
        "| IIB main E[τ] | `data/fig_iib_main_e_tau_summary.csv` | real | same | same | `fig_iib_main_e_tau_hist.{png,svg}` |\n",
        encoding="utf-8",
    )
    print("done")


if __name__ == "__main__":
    main()
