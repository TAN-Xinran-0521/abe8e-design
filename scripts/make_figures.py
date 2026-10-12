# scripts/make_figures.py
# -*- coding: utf-8 -*-
"""
Report-quality figures for the two-locus ABE8e design/off-target comparison.

Story:
    LMNA c.1824 (hard locus)  -> NGG inaccessible (needs VRQR), all candidates
                                 have exploding genome-wide off-targets (1e4-5e4).
    HFE  c.845  (clean locus) -> NGG accessible, candidates have low off-targets
                                 (~600-1000) with a clear gradient -> ranking pays off.

Inputs (auto-detected; missing ones are skipped with a warning):
    PAM accessibility:
        results/lmna_pam_comparison.csv
        results/hfe_pam_comparison.csv
    Reranked candidates (true genome-wide off-target):
        results/offtarget_true/candidates_reranked_true_offtarget.csv       (LMNA)
        results/hfe_offtarget_true/candidates_reranked_true_offtarget.csv    (HFE)

Figures written to --outdir (default results/figures):
    fig1_pam_accessibility.png       grouped bars, LMNA vs HFE, n_candidates per variant
    fig2_efficiency_vs_offtarget.png efficiency vs true off-target, Pareto marked, per locus
    fig3_offtarget_magnitude.png     total off-targets per candidate, log scale, both loci
    fig4_composite_ranking.png       composite score ranking per locus
    fig5_pareto_tradeoff_HFE.png     3-obj bubble (eff/off-target/bystander) for HFE

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --outdir results/figures --root D:/TZS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ----------------------------------------------------------------- styling
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.fontsize": 9,
    "figure.autolayout": True,
})

COLOR_LMNA = "#c0392b"   # red  - hard locus
COLOR_HFE = "#2e86c1"    # blue - clean locus
COLOR_PARETO = "#e67e22"  # orange highlight

VARIANT_ORDER = ["SpCas9", "SpCas9-NG", "SpG", "VRQR", "SpRY"]


# ----------------------------------------------------------------- helpers
def find_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if str(a).lower() in lower:
            return lower[str(a).lower()]
    return None


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing file, skipping: {path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] Empty file, skipping: {path}", file=sys.stderr)
            return None
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
        return None


def short_label(grna: str, n: int = 8) -> str:
    g = str(grna)
    return g if len(g) <= n + 3 else f"{g[:n]}…"


def annotate_bars(ax, bars, fmt="{:.0f}", fontsize=8, dy=0):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h),
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2 + dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize)


# ----------------------------------------------------------------- figure 1
def fig_pam_accessibility(lmna: Optional[pd.DataFrame],
                          hfe: Optional[pd.DataFrame],
                          outpath: Path) -> None:
    if lmna is None and hfe is None:
        print("[WARN] No PAM comparison data; skip fig1.", file=sys.stderr)
        return

    def series_for(df):
        if df is None:
            return {v: np.nan for v in VARIANT_ORDER}
        vcol = find_col(df, ["cas_variant", "variant"])
        ncol = find_col(df, ["n_candidates", "candidates", "count"])
        d = dict(zip(df[vcol].astype(str), pd.to_numeric(df[ncol], errors="coerce")))
        return {v: d.get(v, 0) for v in VARIANT_ORDER}

    lm = series_for(lmna)
    hf = series_for(hfe)

    x = np.arange(len(VARIANT_ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, [lm[v] for v in VARIANT_ORDER], w,
                label="LMNA c.1824 (hard)", color=COLOR_LMNA, alpha=0.9)
    b2 = ax.bar(x + w / 2, [hf[v] for v in VARIANT_ORDER], w,
                label="HFE c.845 (clean)", color=COLOR_HFE, alpha=0.9)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n({p})" for v, p in
                        zip(VARIANT_ORDER, ["NGG", "NGN", "NGN", "NGA", "NRN"])])
    ax.set_ylabel("Accessible candidate sgRNAs")
    ax.set_title("PAM-variant accessibility: no universal PAM\n"
                 "LMNA needs VRQR (NGG=0); HFE works with NGG (VRQR=0)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] {outpath}")


# ----------------------------------------------------------------- figure 2
def fig_efficiency_vs_offtarget(rr_lmna: Optional[pd.DataFrame],
                                rr_hfe: Optional[pd.DataFrame],
                                outpath: Path) -> None:
    panels = [("LMNA c.1824 (hard)", rr_lmna, COLOR_LMNA),
              ("HFE c.845 (clean)", rr_hfe, COLOR_HFE)]
    if all(df is None for _, df, _ in panels):
        print("[WARN] No reranked data; skip fig2.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, df, color) in zip(axes, panels):
        if df is None:
            ax.set_title(f"{title}\n(no data)")
            ax.set_axis_off()
            continue
        eff = pd.to_numeric(df[find_col(df, ["predicted_efficiency", "pred_efficiency"])],
                            errors="coerce")
        risk = pd.to_numeric(df[find_col(df, ["offtarget_risk_norm"])], errors="coerce")
        pcol = find_col(df, ["pareto"])
        pareto = df[pcol].astype(str).str.lower().isin(["true", "1", "1.0"]) \
            if pcol else pd.Series(False, index=df.index)
        gcol = find_col(df, ["grna", "guide"])
        totcol = find_col(df, ["total_offtargets"])

        ax.scatter(risk[~pareto], eff[~pareto], s=90, c=color, alpha=0.55,
                   edgecolor="k", linewidth=0.5, label="dominated")
        ax.scatter(risk[pareto], eff[pareto], s=150, c=COLOR_PARETO,
                   edgecolor="k", linewidth=0.8, marker="D", label="Pareto-optimal")

        for i in df.index:
            lab = short_label(df.loc[i, gcol]) if gcol else str(i)
            if totcol is not None and not pd.isna(df.loc[i, totcol]):
                lab += f"\n({int(df.loc[i, totcol])} OT)"
            ax.annotate(lab, (risk[i], eff[i]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5)

        ax.set_title(title)
        ax.set_xlabel("Normalized genome-wide off-target risk")
        ax.legend(loc="best")
    axes[0].set_ylabel("Predicted editing efficiency")
    fig.suptitle("Efficiency vs true off-target risk (Cas-OFFinder, PAM-agnostic NNN)",
                 fontsize=13, fontweight="bold")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] {outpath}")


# ----------------------------------------------------------------- figure 3
def fig_offtarget_magnitude(rr_lmna: Optional[pd.DataFrame],
                            rr_hfe: Optional[pd.DataFrame],
                            outpath: Path) -> None:
    frames = []
    for locus, df, color in [("LMNA (hard)", rr_lmna, COLOR_LMNA),
                             ("HFE (clean)", rr_hfe, COLOR_HFE)]:
        if df is None:
            continue
        gcol = find_col(df, ["grna", "guide"])
        totcol = find_col(df, ["total_offtargets"])
        if totcol is None:
            continue
        tmp = pd.DataFrame({
            "label": [f"{locus}\n{short_label(g)}" for g in df[gcol]],
            "total": pd.to_numeric(df[totcol], errors="coerce"),
            "color": color,
        })
        frames.append(tmp)
    if not frames:
        print("[WARN] No off-target totals; skip fig3.", file=sys.stderr)
        return

    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(np.arange(len(data)), data["total"].clip(lower=0.5),
                  color=data["color"], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Genome-wide off-target sites (log scale)")
    ax.set_title("Off-target magnitude: hard vs clean locus\n"
                 "LMNA candidates ~10⁴–10⁵ (unusable) vs HFE ~10² (rankable)")
    annotate_bars(ax, bars, fmt="{:.0f}", fontsize=8)
    # legend proxies
    ax.bar(0, 0, color=COLOR_LMNA, label="LMNA c.1824 (hard)")
    ax.bar(0, 0, color=COLOR_HFE, label="HFE c.845 (clean)")
    ax.legend()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] {outpath}")


# ----------------------------------------------------------------- figure 4
def fig_composite_ranking(rr_lmna: Optional[pd.DataFrame],
                          rr_hfe: Optional[pd.DataFrame],
                          outpath: Path) -> None:
    panels = [("LMNA c.1824 (hard)", rr_lmna, COLOR_LMNA),
              ("HFE c.845 (clean)", rr_hfe, COLOR_HFE)]
    if all(df is None for _, df, _ in panels):
        print("[WARN] No reranked data; skip fig4.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, df, color) in zip(axes, panels):
        if df is None:
            ax.set_title(f"{title}\n(no data)")
            ax.set_axis_off()
            continue
        gcol = find_col(df, ["grna", "guide"])
        ccol = find_col(df, ["composite_score_true_offtarget", "composite_score"])
        pcol = find_col(df, ["pareto"])
        d = df.copy()
        d["_score"] = pd.to_numeric(d[ccol], errors="coerce")
        d = d.sort_values("_score", ascending=True)  # horizontal bar: best on top
        pareto = d[pcol].astype(str).str.lower().isin(["true", "1", "1.0"]) \
            if pcol else pd.Series(False, index=d.index)
        colors = [COLOR_PARETO if p else color for p in pareto]
        y = np.arange(len(d))
        ax.barh(y, d["_score"], color=colors, alpha=0.9, edgecolor="k", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([short_label(g, 12) for g in d[gcol]], fontsize=8)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Composite score (w·eff − w·bystander − w·off-target)")
        ax.set_title(title)
    # shared legend
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLOR_PARETO, label="Pareto-optimal"),
                        Patch(color="#888888", label="dominated")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Multi-objective composite ranking (therapy weights 0.4/0.3/0.3)",
                 fontsize=13, fontweight="bold")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {outpath}")


# ----------------------------------------------------------------- figure 5
def fig_pareto_bubble_hfe(rr_hfe: Optional[pd.DataFrame], outpath: Path) -> None:
    if rr_hfe is None:
        print("[WARN] No HFE reranked data; skip fig5.", file=sys.stderr)
        return
    df = rr_hfe.copy()
    eff = pd.to_numeric(df[find_col(df, ["predicted_efficiency", "pred_efficiency"])],
                        errors="coerce")
    risk = pd.to_numeric(df[find_col(df, ["offtarget_risk_norm"])], errors="coerce")
    bcol = find_col(df, ["bystander_a_count", "bystander_A", "bystander"])
    bys = pd.to_numeric(df[bcol], errors="coerce").fillna(0) if bcol else pd.Series(0, index=df.index)
    gcol = find_col(df, ["grna", "guide"])
    vcol = find_col(df, ["cas_variant", "variant"])
    pcol = find_col(df, ["pareto"])
    pareto = df[pcol].astype(str).str.lower().isin(["true", "1", "1.0"]) \
        if pcol else pd.Series(False, index=df.index)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sizes = 200 + bys * 350  # bubble size encodes bystander count
    edge = ["k" if p else "none" for p in pareto]
    lw = [1.6 if p else 0.5 for p in pareto]
    sc = ax.scatter(risk, eff, s=sizes, c=bys, cmap="YlOrRd",
                    edgecolor=edge, linewidth=lw, alpha=0.85, vmin=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Bystander A count (bubble size too)")

    for i in df.index:
        lab = short_label(df.loc[i, gcol], 10)
        if vcol:
            lab += f"\n{df.loc[i, vcol]}"
        if pareto[i]:
            lab += " ★"
        ax.annotate(lab, (risk[i], eff[i]), xytext=(8, 6),
                    textcoords="offset points", fontsize=8)

    ax.set_xlabel("Normalized genome-wide off-target risk")
    ax.set_ylabel("Predicted editing efficiency")
    ax.set_title("HFE c.845: 3-objective trade-off (★ = Pareto-optimal)\n"
                 "highest efficiency ≠ safest choice under therapy weights")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] {outpath}")


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Make report figures for the two-locus comparison.")
    ap.add_argument("--root", default=".", help="Project root (default: current dir).")
    ap.add_argument("--outdir", default="results/figures", help="Figure output directory.")
    ap.add_argument("--lmna_pam", default="results/lmna_pam_comparison.csv")
    ap.add_argument("--hfe_pam", default="results/hfe_pam_comparison.csv")
    ap.add_argument("--lmna_rerank",
                    default="results/offtarget_true/candidates_reranked_true_offtarget.csv")
    ap.add_argument("--hfe_rerank",
                    default="results/hfe_offtarget_true/candidates_reranked_true_offtarget.csv")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    lmna_pam = load_csv(root / args.lmna_pam)
    hfe_pam = load_csv(root / args.hfe_pam)
    lmna_rr = load_csv(root / args.lmna_rerank)
    hfe_rr = load_csv(root / args.hfe_rerank)

    fig_pam_accessibility(lmna_pam, hfe_pam, outdir / "fig1_pam_accessibility.png")
    fig_efficiency_vs_offtarget(lmna_rr, hfe_rr, outdir / "fig2_efficiency_vs_offtarget.png")
    fig_offtarget_magnitude(lmna_rr, hfe_rr, outdir / "fig3_offtarget_magnitude.png")
    fig_composite_ranking(lmna_rr, hfe_rr, outdir / "fig4_composite_ranking.png")
    fig_pareto_bubble_hfe(hfe_rr, outdir / "fig5_pareto_tradeoff_HFE.png")

    print(f"\n[DONE] Figures in: {outdir}")


if __name__ == "__main__":
    main()