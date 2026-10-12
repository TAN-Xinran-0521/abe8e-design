"""
redraw_figC_bystander.py

Redraw figure C for the defense presentation.

Layout:
    Top-left:
        Extra A count within positions 4-8.
        Unit: count.

    Top-right:
        Position-weighted bystander risk score.
        Unit: dimensionless score, range 0-1.

    Bottom:
        Explain the SpRY example using its actual 20-nt sequence.
        Highlight the target A and the non-target A at position 9.

Inputs:
    results/hfe_candidates_bystander.csv
    results/bystander/empirical_prior.json

Outputs:
    results/figures/figC_bystander_count_vs_weighted.png
    results/figures/figC_bystander_count_vs_weighted.svg
    results/figures/figC_bystander_count_vs_weighted.pdf

No model training or inference is performed.
Only existing results are read and checked.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator


# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

COLORS = {
    "text": "#193247",
    "muted": "#64748B",
    "grid": "#E4EAF0",
    "old": "#94A3B8",
    "blue": "#3478B8",
    "blue_light": "#EAF3FB",
    "teal": "#168C82",
    "teal_light": "#E1F3EF",
    "orange": "#D98028",
    "orange_light": "#FFF0DC",
    "border": "#DAE3EC",
    "base": "#F5F7FA",
    "white": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["border"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


# ------------------------------------------------------------
# Data loading and validation
# ------------------------------------------------------------

def find_column(df, names, required=True):
    """Find a column using case-insensitive aliases."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        key = name.lower()
        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Missing required column. Expected one of: {names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_candidates(csv_path):
    df = pd.read_csv(csv_path)

    mapping = {
        "cas_variant": find_column(df, ["cas_variant"]),
        "grna": find_column(df, ["grna", "protospacer"]),
        "count": find_column(df, ["bystander_A", "bystander_a_count"]),
        "weighted": find_column(
            df, ["bystander_weighted", "bystander_ratio"]
        ),
    }

    clean = pd.DataFrame({
        name: df[col] for name, col in mapping.items()
    })

    clean["cas_variant"] = clean["cas_variant"].astype(str).str.strip()
    clean["grna"] = clean["grna"].astype(str).str.strip().str.upper()

    clean["count"] = pd.to_numeric(clean["count"], errors="raise")
    clean["weighted"] = pd.to_numeric(clean["weighted"], errors="raise")

    if clean.empty:
        raise ValueError("Candidate CSV is empty.")

    if not np.isfinite(clean[["count", "weighted"]].to_numpy()).all():
        raise ValueError("Count or weighted score contains NaN/inf.")

    if (clean["count"] < 0).any():
        raise ValueError("Bystander counts must be non-negative.")

    if not np.allclose(clean["count"], np.round(clean["count"])):
        raise ValueError("Bystander counts must be integers.")

    clean["count"] = clean["count"].astype(int)

    if not clean["weighted"].between(0, 1).all():
        raise ValueError("Weighted risk scores must lie between 0 and 1.")

    for seq in clean["grna"]:
        if len(seq) != 20 or not set(seq).issubset(set("ACGT")):
            raise ValueError(f"Expected a valid 20-nt DNA sequence: {seq}")

    # Keep input order. Do not imply a new composite ranking.
    return clean.reset_index(drop=True)


def load_prior(prior_path):
    with open(prior_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    prior = {int(k): float(v) for k, v in raw.items()}

    missing = set(range(1, 21)) - set(prior)
    if missing:
        raise ValueError(f"Prior is missing positions: {sorted(missing)}")

    values = np.array([prior[p] for p in range(1, 21)])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Prior values must be finite and non-negative.")

    return prior


def choose_example(df, variant, requested_sequence=None):
    rows = df[
        df["cas_variant"].str.casefold() == variant.casefold()
    ]

    if requested_sequence:
        rows = rows[
            rows["grna"] == requested_sequence.strip().upper()
        ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one example for variant '{variant}', "
            f"but found {len(rows)}.\n"
            "Use --example-sequence to select a unique candidate."
        )

    return rows.index[0], rows.iloc[0]


def validate_example(
    row,
    prior,
    target_position,
    extra_position,
    window_start,
    window_end,
    tolerance,
):
    seq = row["grna"]

    if not (1 <= window_start <= window_end <= 20):
        raise ValueError("Window positions must satisfy 1 <= start <= end <= 20.")

    if not (window_start <= target_position <= window_end):
        raise ValueError("Target position must lie within the selected window.")

    if not (1 <= extra_position <= 20):
        raise ValueError("Extra-A position must lie between 1 and 20.")

    if target_position == extra_position:
        raise ValueError("Target and extra-A positions must differ.")

    if seq[target_position - 1] != "A":
        raise ValueError(
            f"Position {target_position} is not A in sequence {seq}."
        )

    if seq[extra_position - 1] != "A":
        raise ValueError(
            f"Position {extra_position} is not A in sequence {seq}."
        )

    if window_start <= extra_position <= window_end:
        raise ValueError(
            "The highlighted extra A must be outside the old counting window."
        )

    extra_a_positions = [
        pos
        for pos, base in enumerate(seq, start=1)
        if base == "A" and pos != target_position
    ]

    calculated_count = sum(
        window_start <= pos <= window_end
        for pos in extra_a_positions
    )

    if calculated_count != int(row["count"]):
        raise ValueError(
            "CSV count does not match the sequence/window calculation:\n"
            f"CSV count = {row['count']}, calculated = {calculated_count}."
        )

    target_weight = prior[target_position]
    bystander_weight = sum(prior[pos] for pos in extra_a_positions)
    denominator = target_weight + bystander_weight

    if denominator <= 0:
        raise ValueError("Total target+bystander weight must be positive.")

    calculated_score = bystander_weight / denominator
    saved_score = float(row["weighted"])
    difference = abs(calculated_score - saved_score)

    if difference > tolerance:
        raise ValueError(
            "Saved weighted score differs from the score reconstructed "
            "using the current sequence and prior.\n"
            f"CSV score       = {saved_score:.6f}\n"
            f"Recomputed score= {calculated_score:.6f}\n"
            f"Difference      = {difference:.6f}\n"
            "Check whether the CSV and prior belong to the same run/version."
        )

    if difference > 0.0005:
        warnings.warn(
            "Minor score difference detected, possibly due to rounding. "
            "The chart uses the saved CSV score."
        )

    return {
        "target_weight": target_weight,
        "bystander_weight": bystander_weight,
        "calculated_score": calculated_score,
        "saved_score": saved_score,
        "extra_a_positions": extra_a_positions,
        "count": calculated_count,
    }


# ------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------

def style_chart(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])

    ax.tick_params(axis="both", length=0)


def add_value_labels(ax, bars, values, formatter, offset):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=COLORS["text"],
        )


def add_card(ax, x, y, width, height, facecolor, edgecolor):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1,
        zorder=0,
    )
    ax.add_patch(patch)
    return patch


# ------------------------------------------------------------
# Main figure
# ------------------------------------------------------------

def draw_figure(
    df,
    example_index,
    example,
    prior,
    info,
    target_position,
    extra_position,
    window_start,
    window_end,
):
    # Slide-friendly 16:9 aspect ratio.
    fig = plt.figure(figsize=(16, 9), facecolor="white")

    fig.text(
        0.055, 0.963,
        "From window counts to position-weighted risk",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.text(
        0.055, 0.915,
        "HFE candidates | Different metrics are shown on separate scales",
        fontsize=13,
        color=COLORS["muted"],
        ha="left",
    )

    # Two separate chart panels; no shared y-axis.
    ax_old = fig.add_axes([0.075, 0.525, 0.38, 0.285])
    ax_new = fig.add_axes([0.575, 0.525, 0.38, 0.285])

    x = np.arange(len(df))

    # Sequence prefix helps distinguish candidates without suggesting ranks.
    labels = [
        f"{variant}\n{seq[:8]}..."
        for variant, seq in zip(df["cas_variant"], df["grna"])
    ]

    old_colors = [COLORS["old"]] * len(df)
    new_colors = [COLORS["blue"]] * len(df)

    # Highlight the same example candidate in each panel.
    old_colors[example_index] = COLORS["teal"]
    new_colors[example_index] = COLORS["teal"]

    style_chart(ax_old)
    style_chart(ax_new)

    # -----------------------------
    # Left: count
    # -----------------------------
    counts = df["count"].to_numpy()

    bars_old = ax_old.bar(
        x,
        counts,
        width=0.53,
        color=old_colors,
        zorder=3,
    )

    old_ymax = max(1.4, float(counts.max()) + 0.45)

    ax_old.set_ylim(0, old_ymax)
    ax_old.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_old.set_ylabel("Extra A count")
    ax_old.set_xticks(x)
    ax_old.set_xticklabels(labels, fontsize=11, linespacing=1.5)

    ax_old.set_title(
        "OLD | Window-based count",
        loc="left",
        pad=29,
        color=COLORS["text"],
    )

    ax_old.text(
        0, 1.035,
        f"Non-target A bases at positions {window_start}-{window_end}",
        transform=ax_old.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )

    add_value_labels(
        ax_old,
        bars_old,
        counts,
        formatter=lambda value: str(int(value)),
        offset=old_ymax * 0.035,
    )

    # Make zero-count observations visible at the baseline.
    for xi, value in zip(x, counts):
        if value == 0:
            ax_old.plot(
                xi, 0,
                marker="o",
                markersize=7,
                color=old_colors[xi],
                clip_on=False,
                zorder=4,
            )

    # -----------------------------
    # Right: weighted score
    # -----------------------------
    scores = df["weighted"].to_numpy()

    bars_new = ax_new.bar(
        x,
        scores,
        width=0.53,
        color=new_colors,
        zorder=3,
    )

    ax_new.set_ylim(0, 1.0)
    ax_new.set_yticks(np.linspace(0, 1, 6))
    ax_new.set_ylabel("Position-weighted risk score (0-1)")
    ax_new.set_xticks(x)
    ax_new.set_xticklabels(labels, fontsize=11, linespacing=1.5)

    ax_new.set_title(
        "NEW | Position-weighted score",
        loc="left",
        pad=29,
        color=COLORS["text"],
    )

    ax_new.text(
        0, 1.035,
        "All non-target A bases weighted by the empirical prior",
        transform=ax_new.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )

    add_value_labels(
        ax_new,
        bars_new,
        scores,
        formatter=lambda value: f"{value:.3f}",
        offset=0.025,
    )

    # -----------------------------
    # Bottom explanatory panel
    # -----------------------------
    ax_seq = fig.add_axes([0.055, 0.09, 0.90, 0.345])
    ax_seq.set_xlim(0.3, 20.7)
    ax_seq.set_ylim(0, 3.1)
    ax_seq.axis("off")

    seq = example["grna"]

    ax_seq.text(
        0.01, 0.985,
        f"Why the {example['cas_variant']} example has "
        f"count = {info['count']} but score = {info['saved_score']:.3f}",
        transform=ax_seq.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
    )

    # Numbering description.
    ax_seq.text(
        0.01, 0.86,
        "20-nt protospacer | Position 1 = PAM-distal",
        transform=ax_seq.transAxes,
        fontsize=10.5,
        color=COLORS["muted"],
    )

    # Position labels and nucleotide boxes.
    base_y = 1.49
    base_height = 0.48

    for pos, base in enumerate(seq, start=1):
        face = COLORS["base"]
        edge = COLORS["border"]
        text_color = COLORS["text"]

        if window_start <= pos <= window_end:
            face = COLORS["blue_light"]
            edge = "#BED7ED"

        if base == "A" and pos != target_position:
            face = COLORS["orange_light"]
            edge = COLORS["orange"]

        if pos == target_position:
            face = COLORS["teal"]
            edge = COLORS["teal"]
            text_color = COLORS["white"]

        ax_seq.text(
            pos, 2.12,
            str(pos),
            fontsize=10,
            ha="center",
            color=COLORS["muted"],
        )

        patch = FancyBboxPatch(
            (pos - 0.39, base_y),
            0.78,
            base_height,
            boxstyle="round,pad=0.018,rounding_size=0.055",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.25,
        )
        ax_seq.add_patch(patch)

        ax_seq.text(
            pos,
            base_y + base_height / 2,
            base,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=text_color,
            fontfamily="DejaVu Sans Mono",
        )

    # Bracket marks the old counting window, not the complete biological window.
    bracket_y = 1.29

    ax_seq.plot(
        [window_start - 0.4, window_end + 0.4],
        [bracket_y, bracket_y],
        color=COLORS["blue"],
        linewidth=1.6,
    )

    for endpoint in [window_start - 0.4, window_end + 0.4]:
        ax_seq.plot(
            [endpoint, endpoint],
            [bracket_y, bracket_y + 0.09],
            color=COLORS["blue"],
            linewidth=1.6,
        )

    ax_seq.text(
        (window_start + window_end) / 2,
        1.16,
        f"Old counting window: positions {window_start}-{window_end}",
        fontsize=10,
        ha="center",
        va="top",
        color=COLORS["blue"],
    )

    # Highlight the extra A outside the old window.
    ax_seq.annotate(
        f"Non-target A at position {extra_position}",
        xy=(extra_position, base_y),
        xytext=(13.7, 1.18),
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["orange"],
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["orange"],
            "linewidth": 1.5,
            "connectionstyle": "arc3,rad=-0.12",
        },
    )

    # Two separate explanatory cards.
    add_card(
        ax_seq, 0.01, 0.01, 0.40, 0.22,
        COLORS["teal_light"], "#B9DDD5",
    )

    add_card(
        ax_seq, 0.445, 0.01, 0.54, 0.22,
        COLORS["orange_light"], "#F1D4AE",
    )

    ax_seq.text(
        0.027, 0.175,
        f"Target A: position {target_position}",
        transform=ax_seq.transAxes,
        fontsize=12,
        fontweight="bold",
        color=COLORS["teal"],
    )

    ax_seq.text(
        0.027, 0.07,
        f"Excluded from the extra-A count | "
        f"target weight = {info['target_weight']:.3f}",
        transform=ax_seq.transAxes,
        fontsize=10.5,
        color=COLORS["text"],
    )

    extra_positions = info["extra_a_positions"]

    if extra_positions == [extra_position]:
        formula = (
            f"w{extra_position} / "
            f"(w{target_position} + w{extra_position}) "
            f"= {info['calculated_score']:.3f}"
        )
        detail = (
            f"Position-{extra_position} weight = "
            f"{prior[extra_position]:.3f}"
        )
    else:
        formula = (
            "Sum(non-target A weights) / "
            "(target weight + sum) "
            f"= {info['calculated_score']:.3f}"
        )
        detail = (
            "Non-target A positions: "
            + ", ".join(map(str, extra_positions))
        )

    ax_seq.text(
        0.462, 0.175,
        detail,
        transform=ax_seq.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["orange"],
    )

    ax_seq.text(
        0.462, 0.07,
        formula,
        transform=ax_seq.transAxes,
        fontsize=11,
        color=COLORS["text"],
    )

    # Scientific boundary statement.
    fig.text(
        0.055, 0.035,
        "Interpretation: zero extra A within positions 4-8 does not imply "
        "zero bystander risk. The weighted score is prior-based, "
        "not a measured editing frequency.",
        fontsize=10.5,
        color=COLORS["muted"],
        ha="left",
    )

    return fig


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Redraw figure C with separate scales and a sequence example."
    )

    parser.add_argument("--root", default=__import__('repo_paths').repo_path(''))
    parser.add_argument("--csv", default=None)
    parser.add_argument("--prior", default=None)
    parser.add_argument("--outdir", default=None)

    parser.add_argument("--example-variant", default="SpRY")
    parser.add_argument("--example-sequence", default=None)

    # Explicit coordinates avoid confusing genomic/context positions
    # with 1-based protospacer positions.
    parser.add_argument("--target-position", type=int, default=6)
    parser.add_argument("--extra-position", type=int, default=9)

    parser.add_argument("--window-start", type=int, default=4)
    parser.add_argument("--window-end", type=int, default=8)

    parser.add_argument("--score-tolerance", type=float, default=0.002)
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    root = Path(args.root)

    csv_path = (
        Path(args.csv)
        if args.csv
        else root / "results" / "hfe_candidates_bystander.csv"
    )

    prior_path = (
        Path(args.prior)
        if args.prior
        else root / "results" / "bystander" / "empirical_prior.json"
    )

    outdir = (
        Path(args.outdir)
        if args.outdir
        else root / "results" / "figures"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")

    if not prior_path.is_file():
        raise FileNotFoundError(f"Empirical prior not found: {prior_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    df = load_candidates(csv_path)
    prior = load_prior(prior_path)

    example_index, example = choose_example(
        df,
        variant=args.example_variant,
        requested_sequence=args.example_sequence,
    )

    info = validate_example(
        example,
        prior,
        target_position=args.target_position,
        extra_position=args.extra_position,
        window_start=args.window_start,
        window_end=args.window_end,
        tolerance=args.score_tolerance,
    )

    print("\nExample checked:")
    print(f"  Variant          : {example['cas_variant']}")
    print(f"  Sequence         : {example['grna']}")
    print(f"  Target position  : {args.target_position}")
    print(f"  Non-target A     : {info['extra_a_positions']}")
    print(f"  Old count        : {info['count']}")
    print(f"  CSV score        : {info['saved_score']:.6f}")
    print(f"  Prior-based check: {info['calculated_score']:.6f}")

    fig = draw_figure(
        df,
        example_index,
        example,
        prior,
        info,
        target_position=args.target_position,
        extra_position=args.extra_position,
        window_start=args.window_start,
        window_end=args.window_end,
    )

    stem = outdir / "figC_bystander_count_vs_weighted"

    for suffix in [".png", ".svg", ".pdf"]:
        output_path = stem.with_suffix(suffix)

        if output_path.exists():
            print(f"Overwriting: {output_path}")

        if suffix == ".png":
            fig.savefig(output_path, dpi=args.dpi)
        else:
            fig.savefig(output_path)

        print(f"Saved: {output_path}")

    plt.close(fig)

    print("\nDone. No model was retrained or rerun.")


if __name__ == "__main__":
    main()