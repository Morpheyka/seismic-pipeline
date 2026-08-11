"""
Mean-правдоподобия: распределение E[τ] (mean-only и контраст с mean+range).
Глава: Results (conference_article_ru).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CSV = (
    ROOT
    / "seismic_pipeline_standalone"
    / "artifacts" / "conference_article"
    / "exhaustive_search_parallel.csv"
)
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORT_DIR = ROOT / "reports" / "figures" / "density_safe" / "pub"
LATEX_IMG = Path(__file__).resolve().parents[2] / "latex" / "images"

MEAN_COLORS = {
    "student_t": "#F28E2B",
    "skew_normal": "#4E79A7",
}
MEAN_LABELS = {
    "student_t": "Student-$t$",
    "skew_normal": "skew-normal",
}
PANEL = ["(a)", "(b)", "(c)", "(d)"]
BINS = np.linspace(2.0, 8.0, 25)
TAU_EDGE = 2.05
XLIM = (1.75, 8.15)


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


def shade_prior_edge(ax: plt.Axes) -> None:
    ax.axvspan(2.0, TAU_EDGE, color="#CC3311", alpha=0.12, zorder=0, lw=0)
    ax.axvline(2.0, color="#555555", lw=1.0, ls="--", zorder=3)


def parse_mean_lik(s: str) -> str | None:
    m = re.search(r"mean=([a-z0-9_]+)", str(s))
    return m.group(1) if m else None


def parse_range_lik(s: str) -> str | None:
    m = re.search(r"range=([a-z0-9_]+)", str(s))
    return m.group(1) if m else None


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    feat = df["features"].astype(str)
    df["has_range"] = feat.str.contains(r"\brange\b")
    df["has_mean"] = feat.str.contains(r"\bmean\b")
    df["mean_lik"] = df["likelihoods"].map(parse_mean_lik)
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
    _ensure_dir(REPORT_DIR)
    _ensure_dir(LATEX_IMG)
    _ensure_dir(DATA_DIR)
    for d in (OUT_DIR, REPORT_DIR, LATEX_IMG):
        fig.savefig(d / f"{stem}.png", dpi=450)
        fig.savefig(d / f"{stem}.svg")
    print(f"wrote {stem} → {OUT_DIR}, {REPORT_DIR}, {LATEX_IMG}")


def annotate_box(ax: plt.Axes, x: np.ndarray) -> None:
    med = float(np.median(x))
    frac = edge_frac(pd.Series(x))
    ax.text(
        0.98,
        0.95,
        f"$n$ = {len(x)}\nмедиана = {med:.2f}\nдоля у края = {100 * frac:.0f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.92),
    )
    ax.axvline(med, color="#222222", lw=1.1, alpha=0.8, zorder=3)


def fig_mean_only_by_lik(df: pd.DataFrame) -> None:
    """1×2: все mean-only; наложение student_t vs skew_normal."""
    mo = df[~df["has_range"] & df["has_mean"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

    # (a) pooled
    ax = axes[0]
    x_all = mo["e_tau"].to_numpy()
    ax.hist(
        x_all,
        bins=BINS,
        density=True,
        color="#59A14F",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
    )
    shade_prior_edge(ax)
    annotate_box(ax, x_all)
    ax.set_title(f"{PANEL[0]}  mean-only, все семейства", loc="left", fontweight="bold")
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    ax.set_ylabel("плотность")
    ax.set_xticks([2, 4, 6, 8])

    # (b) by mean lik
    ax = axes[1]
    ymax = 0.0
    for lik in ("student_t", "skew_normal"):
        x = mo.loc[mo["mean_lik"] == lik, "e_tau"].to_numpy()
        h, _ = np.histogram(x, bins=BINS, density=True)
        ymax = max(ymax, float(h.max()) if len(h) else 0.0)
    ymax *= 1.12

    for lik in ("student_t", "skew_normal"):
        x = mo.loc[mo["mean_lik"] == lik, "e_tau"].to_numpy()
        ax.hist(
            x,
            bins=BINS,
            density=True,
            histtype="stepfilled",
            color=MEAN_COLORS[lik],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            label=f"{MEAN_LABELS[lik]} ($n$={len(x)}, мед.={np.median(x):.2f})",
        )
    shade_prior_edge(ax)
    ax.set_title(f"{PANEL[1]}  mean-only по правдоподобию", loc="left", fontweight="bold")
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xticks([2, 4, 6, 8])

    fig.suptitle(
        r"Средний признак без range: $\mathbb{E}[\tau]$ не упирается в prior",
        fontsize=10,
        y=1.06,
    )
    fig.tight_layout()
    save_fig(fig, "fig_mean_only_by_lik")
    plt.close(fig)

    rows = []
    for lik in ("student_t", "skew_normal"):
        x = mo.loc[mo["mean_lik"] == lik, "e_tau"].to_numpy()
        rows.append(
            {
                "mean_lik": lik,
                "n": len(x),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "frac_le_2_05": edge_frac(pd.Series(x)),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "fig_mean_only_by_lik_summary.csv", index=False)


def fig_mean_context(df: pd.DataFrame) -> None:
    """2×2: все с mean vs mean-only vs mean+range (IIB / прочие range)."""
    with_mean = df[df["has_mean"]].copy()
    mo = df[~df["has_range"] & df["has_mean"]]
    mr_iib = df[df["has_mean"] & df["has_range"] & (df["range_lik"] == "interval_inflated_beta")]
    mr_bad = df[
        df["has_mean"]
        & df["has_range"]
        & df["range_lik"].isin(["beta_constrained", "beta", "zero_inflated_beta"])
    ]

    slices = [
        ("все с mean", with_mean["e_tau"].to_numpy(), "#76B7B2"),
        ("mean-only", mo["e_tau"].to_numpy(), "#59A14F"),
        ("mean+range, IIB", mr_iib["e_tau"].to_numpy(), "#4E79A7"),
        ("mean+range, BC/ZOIB/plain", mr_bad["e_tau"].to_numpy(), "#E15759"),
    ]

    ymax = 0.0
    for _, x, _ in slices:
        h, _ = np.histogram(x, bins=BINS, density=True)
        ymax = max(ymax, float(h.max()) if len(h) else 0.0)
    ymax *= 1.12

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True)
    for ax, (title, x, color), letter in zip(axes.ravel(), slices, PANEL):
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
        annotate_box(ax, x)
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", fontsize=8)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks([2, 3, 4, 5, 6, 7, 8])

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    for ax in axes[:, 0]:
        ax.set_ylabel("плотность")

    fig.suptitle(
        r"Контраст: граничный артефакт $\mathbb{E}[\tau]$ появляется при «плохом» range",
        fontsize=10,
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        r"заливка — $\mathbb{E}[\tau]\leq 2.05$; пунктир — $\tau{=}2$; сплошная — медиана",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_fig(fig, "fig_mean_context_e_tau")
    plt.close(fig)


def fig_mean_iib_by_lik(df: pd.DataFrame) -> None:
    """mean+range только IIB: student_t vs skew_normal."""
    sub = df[
        df["has_mean"]
        & df["has_range"]
        & (df["range_lik"] == "interval_inflated_beta")
    ].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ymax = 0.0
    for lik in ("student_t", "skew_normal"):
        x = sub.loc[sub["mean_lik"] == lik, "e_tau"].to_numpy()
        h, _ = np.histogram(x, bins=BINS, density=True)
        ymax = max(ymax, float(h.max()) if len(h) else 0.0)
    ymax *= 1.12

    for lik in ("student_t", "skew_normal"):
        x = sub.loc[sub["mean_lik"] == lik, "e_tau"].to_numpy()
        ax.hist(
            x,
            bins=BINS,
            density=True,
            histtype="stepfilled",
            color=MEAN_COLORS[lik],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            label=f"{MEAN_LABELS[lik]} ($n$={len(x)}, мед.={np.median(x):.2f})",
        )
    shade_prior_edge(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(r"$\mathbb{E}[\tau]$")
    ax.set_ylabel("плотность")
    ax.set_title(r"mean+range, IIB: $\mathbb{E}[\tau]$ по mean-правдоподобию", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_mean_iib_by_lik")
    plt.close(fig)


def update_manifest() -> None:
    man = DATA_DIR.parent / "data-manifest.md"
    extra = (
        "| Mean-only by lik | `data/fig_mean_only_by_lik_summary.csv` | real | same CSV | `plot_e_tau_mean.py` | `fig_mean_only_by_lik.{png,svg}` |\n"
        "| Mean context 2×2 | — | real | same CSV | `plot_e_tau_mean.py` | `fig_mean_context_e_tau.{png,svg}` |\n"
        "| Mean+range IIB by lik | — | real | same CSV | `plot_e_tau_mean.py` | `fig_mean_iib_by_lik.{png,svg}` |\n"
    )
    if man.exists():
        text = man.read_text(encoding="utf-8")
        if "fig_mean_only_by_lik" not in text:
            man.write_text(text.rstrip() + "\n" + extra, encoding="utf-8")


def main() -> None:
    setup_style()
    df = load_frame()
    fig_mean_only_by_lik(df)
    fig_mean_context(df)
    fig_mean_iib_by_lik(df)
    update_manifest()
    print("done")


if __name__ == "__main__":
    main()
