# scripts/extract_context.py
# -*- coding: utf-8 -*-
"""
Extract a real genomic context around a ClinVar variant from hg38, build the
patient (mutant) context, and emit everything design_sgrna.py needs downstream,
including a per-base genomic coordinate mapping for later coordinate-aware
off-target exclusion.

Default target: LMNA c.1824C>T (Hutchinson-Gilford Progeria, HGPS), the canonical
ABE correction target (Koblan et al., Nature 2021).

Biology (why we build a "patient" context):
    - hg38 reference (+ strand) at this locus = C (wild type).
    - Patient carries T (pathogenic c.1824C>T).
    - ABE reverts T:A -> C:G by editing the A on the ANTISENSE strand (A->G).
    - So in the + strand patient context the target position reads 'T', which
      triggers the reverse-strand design path in design_sgrna.py.

Usage (fill in --pos from ClinVar GRCh38 location):
    python scripts/extract_context.py ^
        --genome D:/TZS/genomes/hg38/hg38.fa ^
        --chrom chr1 ^
        --pos 156XXXXXX ^
        --ref C --alt T ^
        --flank 40 ^
        --gene LMNA ^
        --variant_name "LMNA_c.1824C>T_HGPS" ^
        --outdir D:/TZS/data/clinvar

The script will REFUSE to continue if the reference base at --pos is not --ref,
so a wrong coordinate is caught immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from pyfaidx import Fasta
except ImportError:
    print(
        "[ERROR] pyfaidx not installed. Run: pip install pyfaidx",
        file=sys.stderr,
    )
    sys.exit(1)


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract real hg38 context around a ClinVar variant for ABE design."
    )
    p.add_argument("--genome", required=True, help="Path to hg38.fa (UCSC style, >chr1).")
    p.add_argument("--chrom", required=True, help="Chromosome name, e.g. chr1 (must match FASTA header).")
    p.add_argument("--pos", required=True, type=int, help="1-based GRCh38 genomic position of the variant.")
    p.add_argument("--ref", default="C", help="Expected reference base on + strand (default C for LMNA).")
    p.add_argument("--alt", default="T", help="Pathogenic alt base on + strand (default T).")
    p.add_argument("--flank", type=int, default=40, help="Flanking bp on each side (default 40).")
    p.add_argument("--gene", default="LMNA", help="Gene name (metadata only).")
    p.add_argument("--variant_name", default="LMNA_c.1824C>T_HGPS", help="Variant label (metadata only).")
    p.add_argument("--outdir", default="data/clinvar", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ref_base = args.ref.upper()
    alt_base = args.alt.upper()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    genome_path = Path(args.genome)
    if not genome_path.exists():
        print(f"[ERROR] Genome not found: {genome_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading genome (first load builds a .fai index, may take a minute)...")
    genome = Fasta(str(genome_path))

    if args.chrom not in genome:
        available = list(genome.keys())[:10]
        print(
            f"[ERROR] Chromosome '{args.chrom}' not found. "
            f"First few contigs: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    chrom_len = len(genome[args.chrom])
    pos = args.pos  # 1-based
    flank = args.flank

    if pos < 1 or pos > chrom_len:
        print(f"[ERROR] pos {pos} out of range for {args.chrom} (len {chrom_len}).", file=sys.stderr)
        sys.exit(1)

    start_1b = pos - flank            # 1-based start of window
    end_1b = pos + flank              # 1-based end of window (inclusive)
    if start_1b < 1 or end_1b > chrom_len:
        print(f"[ERROR] Window [{start_1b},{end_1b}] out of chromosome bounds.", file=sys.stderr)
        sys.exit(1)

    # pyfaidx is 0-based half-open: [start_1b-1 : end_1b]
    raw = genome[args.chrom][start_1b - 1:end_1b].seq
    wt_context = raw.upper()          # hg38 is soft-masked (lowercase = repeat); normalize

    if len(wt_context) != 2 * flank + 1:
        print(f"[ERROR] Extracted length {len(wt_context)} != expected {2*flank+1}.", file=sys.stderr)
        sys.exit(1)

    target_idx = flank                # 0-based index of the variant within the context
    genome_ref_here = wt_context[target_idx]

    # Repeat-masking note
    is_repeat = any(c.islower() for c in raw)

    print("\n=== Reference (wild-type) context from hg38 ===")
    print(f"  {args.chrom}:{start_1b}-{end_1b}  (+ strand)")
    print(f"  target genomic pos : {pos}")
    print(f"  base at target pos : {genome_ref_here}  (expected ref = {ref_base})")
    print(f"  soft-masked repeat in window: {is_repeat}")
    print(f"  WT  context: {wt_context[:target_idx]}[{genome_ref_here}]{wt_context[target_idx+1:]}")

    # --- Safety net: verify reference base matches expectation ---
    if genome_ref_here != ref_base:
        print(
            f"\n[ERROR] Reference base at {args.chrom}:{pos} is '{genome_ref_here}', "
            f"but you expected '{ref_base}'.\n"
            f"        The coordinate is almost certainly wrong or on the other build/strand.\n"
            f"        Re-check the GRCh38 Location in ClinVar. Aborting.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Build patient (mutant) context: substitute alt at target ---
    patient_context = wt_context[:target_idx] + alt_base + wt_context[target_idx + 1:]
    patient_target_base = patient_context[target_idx]  # == alt_base (T for LMNA)

    print("\n=== Patient (mutant) context ===")
    print(f"  patient context: {patient_context[:target_idx]}[{patient_target_base}]{patient_context[target_idx+1:]}")
    print(f"  target base in patient context: '{patient_target_base}'")
    if patient_target_base == "T":
        print("  -> 'T' at target: design_sgrna.py will scan the REVERSE strand (ABE edits antisense A->G).")
    elif patient_target_base == "A":
        print("  -> 'A' at target: design_sgrna.py will scan the FORWARD strand.")
    else:
        print("  [WARN] target base is not A/T; ABE cannot correct this. Check ref/alt.")

    # --- Per-base genomic coordinate mapping (1-based) ---
    coord_rows = []
    for i, base in enumerate(patient_context):
        coord_rows.append(
            {
                "context_index_0based": i,
                "genomic_pos_1based": start_1b + i,
                "wt_base": wt_context[i],
                "patient_base": base,
                "is_target": (i == target_idx),
            }
        )
    coord_df = pd.DataFrame(coord_rows)

    # --- Write outputs ---
    meta = {
        "variant_name": args.variant_name,
        "gene": args.gene,
        "assembly": "GRCh38/hg38",
        "chrom": args.chrom,
        "target_pos_1based": pos,
        "ref": ref_base,
        "alt": alt_base,
        "strand_note": "coordinates are on + strand; ABE edits the antisense A for a +strand C>T mutation",
        "flank": flank,
        "window_start_1based": start_1b,
        "window_end_1based": end_1b,
        "target_index_in_context_0based": target_idx,
        "wt_context": wt_context,
        "patient_context": patient_context,
        "patient_target_base": patient_target_base,
        "soft_masked_repeat_in_window": is_repeat,
    }

    meta_path = outdir / f"{args.variant_name.replace('>', '_').replace(':', '_')}_meta.json"
    coord_path = outdir / f"{args.variant_name.replace('>', '_').replace(':', '_')}_coord_map.csv"
    patient_fa = outdir / f"{args.variant_name.replace('>', '_').replace(':', '_')}_patient_context.fa"
    wt_fa = outdir / f"{args.variant_name.replace('>', '_').replace(':', '_')}_wt_context.fa"

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    coord_df.to_csv(coord_path, index=False)
    patient_fa.write_text(
        f">{args.variant_name}_patient {args.chrom}:{start_1b}-{end_1b} target={pos}\n{patient_context}\n",
        encoding="utf-8",
    )
    wt_fa.write_text(
        f">{args.variant_name}_wt {args.chrom}:{start_1b}-{end_1b} target={pos}\n{wt_context}\n",
        encoding="utf-8",
    )

    print("\n=== Written ===")
    print(f"  metadata      : {meta_path}")
    print(f"  coord mapping : {coord_path}")
    print(f"  patient fasta : {patient_fa}")
    print(f"  wt fasta      : {wt_fa}")

    print("\n=== Next step ===")
    print("Feed the PATIENT context to your design tool. The target base is at")
    print(f"0-based index {target_idx} of the context (genomic {args.chrom}:{pos}).")
    print("Example (adapt to your design_sgrna.py CLI):")
    print(
        f"  python scripts/design_sgrna.py "
        f"--context {patient_context} --target_pos {target_idx}"
    )


if __name__ == "__main__":
    main()