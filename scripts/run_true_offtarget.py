# scripts/run_true_offtarget.py
# -*- coding: utf-8 -*-
"""
Coordinate-aware genome-wide off-target screening for ABE8e sgRNA candidates.

Key change vs. the earlier version:
    On-target sites are excluded by REAL genomic coordinates
    (ot_chrom / ot_start / ot_end from design_sgrna.py), NOT by the fragile
    "subtract one exact match" heuristic. This is only valid because our
    candidates come from a real ClinVar locus (LMNA c.1824) with known coords.

Cas-OFFinder input format (corrected):
    line 1: genome path
    line 2: pattern = ("N" * 20) + PAM   e.g. NNNNNNNNNNNNNNNNNNNNNGA  (VRQR)
    line 3+: guide_seq + ("N" * len(PAM)) + " " + max_mismatches
    (PAM matching is enforced by the pattern line; guide lines put N for PAM.)

Device:
    This machine exposes only an Intel Arc GPU to Cas-OFFinder, so default
    --device G. Use G0 to pin device id 0 if needed.

Candidate CSV (from design_sgrna.py) is expected to contain:
    grna, pred_efficiency, bystander_A, cas_variant,
    ot_chrom, ot_start, ot_end   (coordinates enable true on-target exclusion)

Outputs (in --outdir):
    casoffinder_input.txt, casoffinder_command.txt, casoffinder_output.txt
    parsed_offtargets.csv
    offtarget_summary_by_grna.csv
    candidates_reranked_true_offtarget.csv
    run_report.md
    plots/*.png
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PRESET_WEIGHTS = {
    "research": (0.8, 0.1, 0.1),
    "therapy": (0.4, 0.3, 0.3),
    "clinical": (0.3, 0.4, 0.3),
}

# Cas variant -> PAM used for the off-target search pattern.
VARIANT_PAM = {
    "SpCas9": "NGG",
    "SpCas9-NG": "NGN",
    "SpG": "NGN",
    "VRQR": "NGA",
    "SpRY": "NRN",
}

COLUMN_ALIASES = {
    "candidate_id": ["candidate_id", "id", "rank", "name"],
    "grna": ["grna", "guide", "sgrna", "protospacer", "spacer"],
    "pam": ["pam", "pam_sequence"],
    "cas_variant": ["cas_variant", "variant", "cas"],
    "strand": ["strand", "target_strand"],
    "predicted_efficiency": [
        "pred_efficiency", "predicted_efficiency", "pred_eff",
        "efficiency", "prediction", "pred",
    ],
    "bystander_A_count": [
        "bystander_a", "bystander_a_count", "bystander_A",
        "bystander", "bystander_count",
    ],
    "ot_chrom": ["ot_chrom", "chrom", "chromosome"],
    "ot_start": ["ot_start", "start"],
    "ot_end": ["ot_end", "end"],
}


# ---------------------------------------------------------------- helpers
def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if str(a).lower() in lower:
            return lower[str(a).lower()]
    return None


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        col = find_column(df, aliases)
        if col is not None and col != canon:
            # do not overwrite an already-correct column
            if canon not in df.columns:
                rename[col] = canon
    return df.rename(columns=rename)


def clean_dna(seq: str) -> str:
    return str(seq).strip().upper().replace("U", "T")


def is_valid_grna(seq: str) -> bool:
    s = clean_dna(seq)
    return len(s) == 20 and set(s).issubset(set("ACGT"))


def safe_int(x) -> Optional[int]:
    try:
        return int(float(str(x).replace(",", "")))
    except Exception:
        return None


IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}


# ---------------------------------------------------------------- read candidates
def read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Candidate file is empty: {path}")

    df = canonicalize(df)

    if "grna" not in df.columns:
        raise ValueError(
            "Candidate CSV must have a guide column (grna/guide/protospacer)."
        )
    df["grna"] = df["grna"].astype(str).map(clean_dna)

    bad = df[~df["grna"].map(is_valid_grna)]
    if len(bad):
        raise ValueError(f"{len(bad)} invalid (non-20nt) guides found:\n{bad[['grna']].head()}")

    if "candidate_id" not in df.columns:
        df.insert(0, "candidate_id", [f"cand_{i+1:03d}" for i in range(len(df))])
    df["candidate_id"] = df["candidate_id"].astype(str)

    if "predicted_efficiency" in df.columns:
        df["predicted_efficiency"] = pd.to_numeric(df["predicted_efficiency"], errors="coerce")
        if df["predicted_efficiency"].max(skipna=True) and df["predicted_efficiency"].max() > 1.5:
            df["predicted_efficiency"] = df["predicted_efficiency"] / 100.0
    else:
        df["predicted_efficiency"] = 0.0

    if "bystander_A_count" in df.columns:
        df["bystander_A_count"] = pd.to_numeric(df["bystander_A_count"], errors="coerce").fillna(0)
    else:
        df["bystander_A_count"] = 0

    # coordinate columns are optional but strongly recommended
    for c in ["ot_chrom", "ot_start", "ot_end"]:
        if c not in df.columns:
            df[c] = np.nan
    df["ot_start"] = pd.to_numeric(df["ot_start"], errors="coerce")
    df["ot_end"] = pd.to_numeric(df["ot_end"], errors="coerce")

    has_coords = df["ot_chrom"].notna().any() and df["ot_start"].notna().any()
    if not has_coords:
        print(
            "[WARN] No genomic coordinates (ot_chrom/ot_start/ot_end) in candidates.\n"
            "       On-target exclusion will fall back to 'closest exact match'.\n"
            "       Provide coordinates via design_sgrna.py --chrom --context_start.",
            file=sys.stderr,
        )
    return df


def resolve_search_pam(df: pd.DataFrame, override: Optional[str]) -> str:
    if override:
        return override.upper()
    if "cas_variant" in df.columns:
        variants = {str(v).split("(")[0] for v in df["cas_variant"].dropna().unique()}
        pams = {VARIANT_PAM.get(v) for v in variants if v in VARIANT_PAM}
        pams.discard(None)
        if len(pams) == 1:
            return pams.pop()
        if len(pams) > 1:
            raise ValueError("Mixed PAM rules: split candidates by variant before searching.")
            # Mixed-PAM input is rejected above.
    raise ValueError("Cannot infer PAM. Provide a recognized variant or explicit --pam.")
    return "NGG"


# ---------------------------------------------------------------- Cas-OFFinder I/O
def write_casoffinder_input(
    candidates: pd.DataFrame,
    genome_path: Path,
    input_path: Path,
    pam: str,
    max_mismatches: int,
) -> pd.DataFrame:
    pam = pam.upper()
    pattern = "N" * 20 + pam            # e.g. NNNNNNNNNNNNNNNNNNNNNGA
    pam_placeholder = "N" * len(pam)    # guide lines put N for the PAM positions

    guides = candidates[["grna"]].drop_duplicates().sort_values("grna").reset_index(drop=True)

    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(str(genome_path).replace("\\", "/") + "\n")
        f.write(pattern + "\n")
        for g in guides["grna"]:
            f.write(f"{g}{pam_placeholder} {max_mismatches}\n")
    return guides


def build_command(exe, input_path, output_path, device) -> List[str]:
    return [str(exe), str(input_path), device, str(output_path)]


def run_casoffinder(exe, input_path, output_path, device) -> None:
    cmd = build_command(exe, input_path, output_path, device)
    print("[INFO] Running Cas-OFFinder:\n       " + subprocess.list2cmdline(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cas-OFFinder exe not found: {exe}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Cas-OFFinder failed (code {e.returncode}). Cmd: {subprocess.list2cmdline(cmd)}")


def is_dna_like(s: str) -> bool:
    s = re.sub(r"[^ACGTNRYSWKMBDHV\-]", "", str(s).upper())
    return len(s) >= 20


def parse_casoffinder_output(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        raise FileNotFoundError(f"Cas-OFFinder output not found: {output_path}")
    rows = []
    with output_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = re.split(r"\s+", raw)
            if len(parts) < 6:
                continue
            if parts[0].lower() in {"crrna", "guide", "query", "id"}:
                continue
            query = parts[0].upper().replace("U", "T")
            # two common column orders
            if is_dna_like(parts[1]):
                offseq, chrom, pos, strand, mm = parts[1], parts[2], safe_int(parts[3]), parts[4], safe_int(parts[5])
            else:
                chrom, pos, offseq, strand, mm = parts[1], safe_int(parts[2]), parts[3], parts[4], safe_int(parts[5])
            if mm is None or pos is None:
                continue
            guide = query[:20]
            if len(guide) != 20:
                continue
            rows.append({
                "grna": guide,
                "chrom": chrom,
                "position": pos,
                "offtarget_sequence": str(offseq).upper().replace("U", "T"),
                "strand": strand,
                "mismatches": int(mm),
                "raw_line": raw,
            })
    cols = ["grna", "chrom", "position", "offtarget_sequence", "strand", "mismatches", "raw_line"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------- risk scoring
def mismatch_positions_1based(guide20: str, off20: str) -> List[int]:
    g, o = clean_dna(guide20)[:20], clean_dna(off20)[:20]
    if len(g) != 20 or len(o) != 20:
        return []
    return [i for i, (a, b) in enumerate(zip(g, o), start=1) if a != b]


def position_penalty(pos: int) -> float:
    # PAM-distal (1-8) more tolerant of mismatch -> higher residual risk;
    # PAM-proximal seed (13-20) mismatch is more disruptive -> lower residual risk.
    if 1 <= pos <= 8:
        return 0.75
    if 9 <= pos <= 12:
        return 0.50
    return 0.25


def fallback_mm_weight(mm: int) -> float:
    return {0: 1.0, 1: 0.6, 2: 0.25, 3: 0.08, 4: 0.02, 5: 0.005}.get(mm, 0.001)


def per_hit_risk(row: pd.Series) -> float:
    mm = int(row["mismatches"])
    if mm <= 0:
        return 1.0
    positions = mismatch_positions_1based(row["grna"], row["offtarget_sequence"])
    if not positions:
        return fallback_mm_weight(mm)
    score = 1.0
    for p in positions:
        score *= position_penalty(p)
    if len(positions) < mm:
        score *= 0.5 ** (mm - len(positions))
    return float(score)


# ---------------------------------------------------------------- coordinate-aware on-target exclusion
def mark_on_target(offtargets, candidates, tolerance=0, pam_length=3):
    """Exact intended-target exclusion for tested 3-prime PAM, no-bulge output.

    Cas-OFFinder position is the 0-based leftmost position of the full
    guide+PAM match. A reference/variant mismatch does not disqualify on-target.
    """
    if tolerance != 0:
        raise ValueError("Broad coordinate tolerance is disabled; use exact matching.")
    required = ["ot_chrom", "ot_start", "ot_end", "strand"]
    if any(c not in candidates for c in required):
        raise ValueError("Exact exclusion requires chromosome, span and strand.")
    if candidates[required].isna().any().any():
        raise ValueError("Every candidate needs complete intended-target coordinates.")
    if candidates["grna"].duplicated().any():
        raise ValueError(
            "This parser requires one intended target per guide per run; "
            "split repeated-guide candidate instances before exclusion."
        )

    df = offtargets.copy()
    df["is_on_target"] = False
    if df.empty:
        return df

    if not df["strand"].isin(["+", "-"]).all():
        raise ValueError("Unexpected hit strand.")
    pos = pd.to_numeric(df["position"], errors="raise")
    shift = np.where(df["strand"].eq("-"), pam_length, 0)
    df["hit_proto_start_1based"] = pos + shift + 1
    df["hit_proto_end_1based"] = df["hit_proto_start_1based"] + 19

    intended = candidates.set_index("grna")
    def check(row):
        if row["grna"] not in intended.index:
            raise ValueError("Search output contains a guide absent from input.")
        target = intended.loc[row["grna"]]
        return (
            str(row["chrom"]) == str(target["ot_chrom"])
            and row["strand"] == target["strand"]
            and row["hit_proto_start_1based"] == target["ot_start"]
            and row["hit_proto_end_1based"] == target["ot_end"]
        )
    df["is_on_target"] = df.apply(check, axis=1)
    return df

def summarize(offtargets: pd.DataFrame, max_mm: int) -> pd.DataFrame:
    base_cols = (
        ["grna", "total_offtargets", "on_target_hits", "offtarget_risk_raw"]
        + [f"hit_mm{m}" for m in range(max_mm + 1)]
    )
    if offtargets.empty:
        return pd.DataFrame(columns=base_cols)

    df = offtargets.copy()
    df["hit_score"] = df.apply(per_hit_risk, axis=1)

    off = df[~df["is_on_target"]].copy()   # true off-targets only
    on = df[df["is_on_target"]].copy()

    agg = off.groupby("grna").agg(
        total_offtargets=("grna", "size"),
        offtarget_risk_raw=("hit_score", "sum"),
    )
    on_counts = on.groupby("grna").size().rename("on_target_hits")

    mm_counts = pd.crosstab(off["grna"], off["mismatches"]) if len(off) else pd.DataFrame()
    for m in range(max_mm + 1):
        if m not in mm_counts.columns:
            mm_counts[m] = 0
    mm_counts = mm_counts[[m for m in range(max_mm + 1)]]
    mm_counts.columns = [f"hit_mm{m}" for m in range(max_mm + 1)]

    summary = agg.join(mm_counts, how="outer").join(on_counts, how="outer").fillna(0).reset_index()
    # guides with zero off-targets but present on-target still appear
    all_guides = pd.DataFrame({"grna": df["grna"].unique()})
    summary = all_guides.merge(summary, on="grna", how="left").fillna(0)

    int_cols = ["total_offtargets", "on_target_hits"] + [f"hit_mm{m}" for m in range(max_mm + 1)]
    for c in int_cols:
        summary[c] = summary[c].astype(int)
    return summary[base_cols]


# ---------------------------------------------------------------- rerank
def norm_log01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    if s.max() <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return np.log1p(s) / math.log1p(float(s.max()))


def pareto(df, eff, bys, risk) -> pd.Series:
    a = df[[eff, bys, risk]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    n = len(a)
    keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            no_worse = a[j][0] >= a[i][0] and a[j][1] <= a[i][1] and a[j][2] <= a[i][2]
            better = a[j][0] > a[i][0] or a[j][1] < a[i][1] or a[j][2] < a[i][2]
            if no_worse and better:
                keep[i] = False
                break
    return pd.Series(keep, index=df.index)


def rerank(candidates, summary, scenario, w=(None, None, None), bys_norm=4.0) -> pd.DataFrame:
    df = candidates.merge(summary, on="grna", how="left")
    for c in ["total_offtargets", "on_target_hits", "offtarget_risk_raw"] + \
             [x for x in df.columns if re.match(r"hit_mm\d+", str(x))]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "offtarget_risk_raw" not in df.columns:
        df["offtarget_risk_raw"] = 0.0

    df["offtarget_risk_norm"] = norm_log01(df["offtarget_risk_raw"])
    df["predicted_efficiency"] = pd.to_numeric(df["predicted_efficiency"], errors="coerce").fillna(0).clip(0, 1)
    df["bystander_A_count"] = pd.to_numeric(df["bystander_A_count"], errors="coerce").fillna(0).clip(lower=0)
    if "bystander_weighted" not in df.columns:
        raise ValueError(
            "Missing bystander_weighted. Regenerate candidates with the empirical prior; "
            "historical window-count reranking is not silently substituted."
        )
    by = pd.to_numeric(df["bystander_weighted"], errors="raise")
    if not np.isfinite(by).all() or not by.between(0, 1).all():
        raise ValueError("bystander_weighted must be finite and in [0,1].")
    df["bystander_norm"] = by

    we, wb, wo = w
    if None in w:
        pe, pb, po = PRESET_WEIGHTS[scenario]
        we, wb, wo = (pe if we is None else we), (pb if wb is None else wb), (po if wo is None else wo)
    df["weight_eff"], df["weight_bystander"], df["weight_offtarget"] = we, wb, wo

    df["composite_score_true_offtarget"] = (
        we * df["predicted_efficiency"]
        - wb * df["bystander_norm"]
        - wo * df["offtarget_risk_norm"]
    )
    df["pareto"] = pareto(df, "predicted_efficiency", "bystander_norm", "offtarget_risk_norm")
    df = df.sort_values(
        ["pareto", "composite_score_true_offtarget", "predicted_efficiency", "offtarget_risk_norm"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    df.insert(0, "rank_true_offtarget", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------- plots + report
def make_plots(reranked: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib missing; skipping plots.", file=sys.stderr)
        return
    pd_ = outdir / "plots"
    pd_.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    colors = reranked["pareto"].map({True: "tab:red", False: "tab:blue"})
    plt.scatter(reranked["offtarget_risk_norm"], reranked["predicted_efficiency"],
                c=colors, edgecolor="k", linewidth=0.4, alpha=0.85)
    plt.xlabel("Normalized genome-wide off-target risk")
    plt.ylabel("Predicted editing efficiency")
    plt.title("Efficiency vs true off-target risk (Cas-OFFinder)")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(pd_ / "efficiency_vs_true_offtarget.png", dpi=300); plt.close()

    top = reranked.head(min(15, len(reranked)))
    plt.figure(figsize=(9, 5))
    plt.bar(top["candidate_id"].astype(str), top["composite_score_true_offtarget"], color="tab:green", alpha=0.8)
    plt.xticks(rotation=60, ha="right"); plt.ylabel("Composite score")
    plt.title("Top candidates after true off-target reranking")
    plt.grid(axis="y", alpha=0.25); plt.tight_layout()
    plt.savefig(pd_ / "top_candidates_composite_score.png", dpi=300); plt.close()


def write_report(outdir, args, candidates, guides, offtargets, marked, summary, reranked, pam, cmd):
    cols = ["rank_true_offtarget", "candidate_id", "grna", "cas_variant",
            "predicted_efficiency", "bystander_A_count", "total_offtargets",
            "offtarget_risk_norm", "composite_score_true_offtarget", "pareto"]
    cols = [c for c in cols if c in reranked.columns]
    lines = [
        "# True (coordinate-aware) Off-target Reranking Report", "",
        "## Input",
        f"- Candidates: `{args.candidates}`",
        f"- Genome: `{args.genome}`",
        f"- Search PAM: `{pam}`  |  Max mismatches: `{args.max_mismatches}`",
        f"- Scenario: `{args.scenario}`  |  Device: `{args.device}`",
        f"- On-target exclusion: coordinate-based (tolerance {args.tolerance} bp)"
        if candidates['ot_start'].notna().any() else
        "- On-target exclusion: fallback (no coordinates provided)",
        "", "## Cas-OFFinder command", "```",
        subprocess.list2cmdline(cmd) if cmd else "(parsed existing output)", "```",
        "", "## Summary",
        f"- Candidate rows: {len(candidates)}",
        f"- Unique guides: {len(guides)}",
        f"- Parsed hits: {len(offtargets)}",
        f"- On-target hits removed: {int(marked['is_on_target'].sum()) if not marked.empty else 0}",
        f"- True off-target hits: {int((~marked['is_on_target']).sum()) if not marked.empty else 0}",
        "", "## Top candidates (coordinate-aware true off-target)", "",
        reranked[cols].head(10).to_string(index=False) if not reranked.empty else "_none_",
        "", "## Notes",
        "- On-target sites excluded by real genomic coordinates from a ClinVar locus (LMNA c.1824), not by heuristic.",
        "- `offtarget_risk_raw` = mismatch-count + position weighted aggregate of true off-target hits.",
        "- This is genome-wide **Cas9-dependent** off-target screening only.",
        "  Cas9-independent deaminase off-target (random ssDNA/RNA editing) is NOT captured; needs GOTI/Detect-seq.",
        "- PAM-relaxed variants (VRQR/SpRY) trade wider targeting for potentially more off-targets.",
    ]
    (outdir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- args + main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coordinate-aware Cas-OFFinder off-target reranking for ABE8e candidates.")
    p.add_argument("--candidates", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--casoffinder_exe", default="cas-offinder")
    p.add_argument("--outdir", default="results/offtarget_true")
    p.add_argument("--pam", default=None, help="Override search PAM (IUPAC). Default: inferred from cas_variant.")
    p.add_argument("--max_mismatches", type=int, default=4)
    p.add_argument("--device", default="G", help="Cas-OFFinder device: G (GPU, default here), G0, or C (CPU).")
    p.add_argument("--scenario", default="therapy", choices=list(PRESET_WEIGHTS.keys()))
    p.add_argument("--w_eff", type=float, default=None)
    p.add_argument("--w_bys", type=float, default=None)
    p.add_argument("--w_off", type=float, default=None)
    p.add_argument("--bystander_norm", type=float, default=4.0)
    p.add_argument("--tolerance", type=int, default=0, help="Exact matching only; must be 0.")
    p.add_argument("--cas_output", default=None, help="Parse an existing Cas-OFFinder output instead of running.")
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--summary_only", action="store_true",
                   help="Stop after hit summary; do not perform subgroup ranking.")
    p.add_argument("--dry_run", action="store_true", help="Only write Cas-OFFinder input + command.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = outdir / "casoffinder_input.txt"
    output_path = outdir / "casoffinder_output.txt"

    print("[INFO] Reading candidates...")
    candidates = read_candidates(Path(args.candidates))
    candidates.to_csv(outdir / "candidates_normalized_input.csv", index=False)

    pam = resolve_search_pam(candidates, args.pam)
    print(f"[INFO] Search PAM: {pam}")

    print("[INFO] Writing Cas-OFFinder input...")
    guides = write_casoffinder_input(candidates, Path(args.genome), input_path, pam, args.max_mismatches)
    cmd = build_command(args.casoffinder_exe, input_path, output_path, args.device)
    (outdir / "casoffinder_command.txt").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"[DRY RUN] input -> {input_path}\n[DRY RUN] cmd   -> {subprocess.list2cmdline(cmd)}")
        return

    if args.cas_output:
        output_path = Path(args.cas_output); cmd_for_report = None
        print(f"[INFO] Parsing existing output: {output_path}")
    elif args.skip_run:
        cmd_for_report = None
        print(f"[INFO] Skipping run; parsing {output_path}")
    else:
        run_casoffinder(args.casoffinder_exe, input_path, output_path, args.device)
        cmd_for_report = cmd

    print("[INFO] Parsing Cas-OFFinder output...")
    offtargets = parse_casoffinder_output(output_path)
    offtargets.to_csv(outdir / "parsed_offtargets.csv", index=False)

    print("[INFO] Marking on-target hits by genomic coordinates...")
    marked = mark_on_target(offtargets, candidates, tolerance=args.tolerance, pam_length=len(pam))
    marked.to_csv(outdir / "parsed_offtargets_marked.csv", index=False)

    print("[INFO] Summarizing true off-targets...")
    summary = summarize(marked, args.max_mismatches)
    summary.to_csv(outdir / "offtarget_summary_by_grna.csv", index=False)

    if args.summary_only:
        print("[DONE] Search and exact-coordinate summary complete.")
        return

    print("[INFO] Reranking...")
    reranked = rerank(candidates, summary, args.scenario,
                      w=(args.w_eff, args.w_bys, args.w_off), bys_norm=args.bystander_norm)
    reranked.to_csv(outdir / "candidates_reranked_true_offtarget.csv", index=False)

    make_plots(reranked, outdir)
    write_report(outdir, args, candidates, guides, offtargets, marked, summary, reranked, pam, cmd_for_report)

    print(f"\n[DONE] Output dir: {outdir}")
    show = ["rank_true_offtarget", "candidate_id", "grna", "predicted_efficiency",
            "total_offtargets", "offtarget_risk_norm", "composite_score_true_offtarget", "pareto"]
    show = [c for c in show if c in reranked.columns]
    print(reranked[show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()