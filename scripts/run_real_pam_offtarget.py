#!/usr/bin/env python3
"""Wrapper: run true-PAM off-target per cas_variant, then merge vs NNN upper bound.
Splits candidates by cas_variant (each subfile -> run_true_offtarget.py auto-resolves
the variant's REAL PAM), runs Cas-OFFinder per subfile, and builds a comparison table:
  candidate | grna | cas_variant | real_total_offtargets | nnn_total_offtargets
"""
import argparse, os, subprocess, sys
import pandas as pd

from pathlib import Path
from run_true_offtarget import read_candidates, rerank

SCRIPT = str(Path(__file__).resolve().with_name("run_true_offtarget.py"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--casoffinder_exe", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--device", default="G")
    ap.add_argument("--max_mismatches", type=int, default=4)
    ap.add_argument("--nnn_summary", default=None,
                    help="Optional path to prior NNN offtarget_summary_by_grna.csv for comparison.")
    ap.add_argument("--scenario", choices=["research", "therapy", "clinical"],
                    default="therapy")
    args = ap.parse_args()

    os.makedirs(args.outroot, exist_ok=True)
    df = read_candidates(Path(args.candidates))
    if "bystander_weighted" not in df.columns:
        raise ValueError("Input must contain empirical bystander_weighted scores.")
    if df["grna"].duplicated().any():
        raise ValueError("This release requires unique guides in the merged input.")
    if "cas_variant" not in df.columns:
        sys.exit("[ERR] candidates need a cas_variant column.")

    merged = []
    for variant, sub in df.groupby("cas_variant"):
        vtag = str(variant).replace("-", "").replace("(", "_").replace(")", "")
        subfile = os.path.join(args.outroot, f"cand_{vtag}.csv")
        outdir  = os.path.join(args.outroot, f"run_{vtag}")
        sub.to_csv(subfile, index=False)
        cmd = [sys.executable, SCRIPT,
               "--candidates", subfile,
               "--genome", args.genome,
               "--casoffinder_exe", args.casoffinder_exe,
               "--outdir", outdir,
               "--device", args.device,
               "--max_mismatches", str(args.max_mismatches), "--summary_only"]
        print(f"\n=== variant {variant}: real PAM run ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        spath = os.path.join(outdir, "offtarget_summary_by_grna.csv")
        if os.path.exists(spath):
            s = pd.read_csv(spath).copy()
            s["cas_variant"] = variant
            merged.append(s)

    if not merged:
        sys.exit("[ERR] no summaries produced.")
    combined_summary = pd.concat(merged, ignore_index=True)
    missing = set(df["grna"]) - set(combined_summary["grna"])
    if missing:
        raise ValueError(
            f"Missing summary for guides {missing}; verify zero-hit status explicitly."
        )
    final_ranked = rerank(
        df,
        combined_summary.drop(columns=["cas_variant"]),
        args.scenario
    )
    final_ranked.to_csv(
        os.path.join(args.outroot, "candidates_ranked_global.csv"), index=False
    )

    real = pd.concat(merged, ignore_index=True).rename(
        columns={"total_offtargets": "real_total_offtargets"})

    # attach efficiency/id from candidates
    keep = [c for c in ["candidate_id", "rank", "grna", "cas_variant",
                        "pred_efficiency", "predicted_efficiency"] if c in df.columns]
    out = df[keep].merge(real, on=["grna", "cas_variant"], how="left")

    if args.nnn_summary and os.path.exists(args.nnn_summary):
        nnn = pd.read_csv(args.nnn_summary)[["grna", "total_offtargets"]].rename(
            columns={"total_offtargets": "nnn_total_offtargets"})
        out = out.merge(nnn, on="grna", how="left")
        out["real_over_nnn_pct"] = (out["real_total_offtargets"] /
                                    out["nnn_total_offtargets"] * 100).round(1)

    outfile = os.path.join(args.outroot, "real_vs_nnn_comparison.csv")
    out.to_csv(outfile, index=False)
    print("\n=== Real-PAM vs NNN comparison ===")
    print(out.to_string(index=False))
    print(f"\nWritten -> {outfile}")

if __name__ == "__main__":
    main()
