#!/usr/bin/env python3
"""
Redraw the two-site potential off-target comparison.

One subplot only:
    - Two bars: HFE and LMNA.
    - Linear y-axis starting at zero.
    - Count labels above bars.
    - Site and Cas-variant labels on the x-axis.
    - No title, interpretation, efficiency annotations, or ratio.

Candidate selection:
    Select the candidate with the lowest real_total_offtargets
    within each supplied CSV, matching the original script.
    This is NOT a claim of overall optimality or experimental safety.

Inputs:
    results/hfe_offtarget_real/real_vs_nnn_comparison.csv
    results/lmna_offtarget_real/real_vs_nnn_comparison.csv

Outputs:
    results/figures/fig_two_site_realpam.png
    results/figures/fig_two_site_realpam.svg
    results/figures/fig_two_site_realpam.pdf
    results/figures/fig_two_site_realpam_selected_candidates.csv

No models or off-target searches are rerun.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "axes.edgecolor": "#AAB5C2",
    "axes.labelcolor": "#23374D",
    "xtick.color": "#23374D",
    "ytick.color": "#526174",
    "text.color": "#23374D",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def load_candidates(path):
    """Read and validate an existing per-site comparison table."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)

    required = [
        "cas_variant",
        "grna",
        "real_total_offtargets",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    df = df.copy()

    if df[["cas_variant", "grna"]].isna().any().any():
        raise ValueError(f"Missing variant or gRNA values in: {path}")

    df["cas_variant"] = df["cas_variant"].astype(str).str.strip()
    df["grna"] = df["grna"].astype(str).str.strip().str.upper()

    if (df["cas_variant"] == "").any() or (df["grna"] == "").any():
        raise ValueError(f"Empty variant or gRNA values in: {path}")

    counts = pd.to_numeric(
        df["real_total_offtargets"], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(counts).all():
        raise ValueError(f"Off-target counts contain NaN/inf in: {path}")

    if (counts < 0).any():
        raise ValueError(f"Off-target counts must be non-negative: {path}")

    if not np.allclose(counts, np.round(counts)):
        raise ValueError(f"Off-target counts must be integers: {path}")

    df["real_total_offtargets"] = np.round(counts).astype(np.int64)

    return df


def select_candidate(df, site, source_path):
    """
    Select the minimum-count candidate.
    If tied, retain the first matching row in the input CSV,
    consistent with the original idxmin behavior.
    """
    minimum = df["real_total_offtargets"].min()
    tied = df[df["real_total_offtargets"] == minimum]

    if len(tied) > 1:
        print(
            f"Warning: {site} has {len(tied)} candidates tied at "
            f"{minimum} potential sites; selecting the first CSV row."
        )

    row = tied.iloc[0]

    result = {
        "site": site,
        "cas_variant": row["cas_variant"],
        "grna": row["grna"],
        "predicted_potential_off_target_sites": int(
            row["real_total_offtargets"]
        ),
        "selection_rule": "minimum count within supplied site table",
        "source_csv": str(source_path),
    }

    # Preserve optional metadata for audit, not for plotting.
    if "pred_efficiency" in df.columns:
        result["pred_efficiency"] = row["pred_efficiency"]

    if "rank" in df.columns:
        result["input_rank"] = row["rank"]

    return result


def draw_chart(selected, outdir, dpi):
    counts = selected[
        "predicted_potential_off_target_sites"
    ].to_numpy(dtype=int)

    labels = [
        f"{row['site']}\n{row['cas_variant']}"
        for _, row in selected.iterrows()
    ]

    # Same neutral color: no green/red safety implication.
    bar_color = "#477FA8"

    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor="white")

    x = np.arange(len(selected))

    bars = ax.bar(
        x,
        counts,
        width=0.48,
        color=bar_color,
        edgecolor="none",
        zorder=3,
    )

    # Linear scale with a true zero baseline.
    max_count = int(counts.max())
    ymax = max(5.0, max_count * 1.18)

    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.65, len(selected) - 0.35)

    ax.set_ylabel(
        "Predicted potential off-target sites",
        labelpad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=1.7)

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=True)
    )
    ax.yaxis.set_major_formatter(
        StrMethodFormatter("{x:,.0f}")
    )

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color="#E2E8EF",
        linewidth=0.8,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=8)

    # Data only above bars: no efficiency, ratio, or interpretation.
    for bar, count in zip(bars, counts):
        ax.annotate(
            f"{count:,}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color="#23374D",
        )

    # Intentionally no title, legend, caption or textbox.
    fig.subplots_adjust(
        left=0.17,
        right=0.98,
        bottom=0.21,
        top=0.96,
    )

    stem = outdir / "fig_two_site_realpam"

    for extension in [".png", ".svg", ".pdf"]:
        output = stem.with_suffix(extension)

        if output.exists():
            print(f"Overwriting: {output}")

        if extension == ".png":
            fig.savefig(output, dpi=dpi)
        else:
            fig.savefig(output)

        print(f"Saved: {output}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Redraw a single-panel HFE vs LMNA comparison."
    )

    parser.add_argument(
        "--root",
        default=__import__('repo_paths').repo_path(''),
        help="Project root directory.",
    )
    parser.add_argument(
        "--hfe-csv",
        default=None,
        help="Optional HFE CSV path.",
    )
    parser.add_argument(
        "--lmna-csv",
        default=None,
        help="Optional LMNA CSV path.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )

    args = parser.parse_args()
    root = Path(args.root)

    hfe_path = (
        Path(args.hfe_csv)
        if args.hfe_csv
        else root
        / "results"
        / "hfe_offtarget_real"
        / "real_vs_nnn_comparison.csv"
    )

    lmna_path = (
        Path(args.lmna_csv)
        if args.lmna_csv
        else root
        / "results"
        / "lmna_offtarget_real"
        / "real_vs_nnn_comparison.csv"
    )

    outdir = (
        Path(args.outdir)
        if args.outdir
        else root / "results" / "figures"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    hfe = load_candidates(hfe_path)
    lmna = load_candidates(lmna_path)

    selected = pd.DataFrame([
        select_candidate(
            hfe,
            site="HFE c.845G>A",
            source_path=hfe_path,
        ),
        select_candidate(
            lmna,
            site="LMNA c.1824C>T",
            source_path=lmna_path,
        ),
    ])

    # Save the actual plotted data and candidate provenance.
    selected_path = (
        outdir / "fig_two_site_realpam_selected_candidates.csv"
    )
    selected.to_csv(
        selected_path,
        index=False,
        encoding="utf-8-sig",
    )

    draw_chart(selected, outdir, dpi=args.dpi)

    print("\nSelected candidates:")
    print(
        selected[
            [
                "site",
                "cas_variant",
                "grna",
                "predicted_potential_off_target_sites",
            ]
        ].to_string(index=False)
    )

    print(f"\nPlotted data saved to: {selected_path}")
    print("Done. No models or off-target searches were rerun.")


if __name__ == "__main__":
    main()