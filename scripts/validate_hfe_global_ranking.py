"""Reanalyse saved HFE search outputs and rank all three candidates together.

No model training and no Cas-OFFinder execution.
This checks saved-output consistency, not genome-search completeness.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_true_offtarget import (
    read_candidates,
    parse_casoffinder_output,
    mark_on_target,
    summarize,
    rerank,
)

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "SpCas9": {
        "directory": "run_SpCas9",
        "pam": "NGG",
    },
    "SpCas9-NG": {
        "directory": "run_SpCas9NG",
        "pam": "NGN",
    },
    "SpRY": {
        "directory": "run_SpRY",
        "pam": "NRN",
    },
}

IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
    "N": set("ACGT"), "R": set("AG"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_saved_input(path, guide, pam):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    require(len(lines) == 3, f"Expected exactly one query in {path}")
    require(lines[1] == "N" * 20 + pam, f"PAM pattern mismatch: {path}")

    query = lines[2].split()
    require(len(query) == 2, f"Unexpected query format: {path}")
    require(query[0] == guide + "N" * len(pam), f"Guide mismatch: {path}")
    require(int(query[1]) == 4, f"Expected max_mismatches=4: {path}")
    return lines[0]


def validate_parsed_hits(path, hits, guide, pam):
    # Known saved-output format has no header. Reject silently skipped lines.
    raw_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(
        len(raw_lines) == len(hits),
        f"Raw/parsed line count mismatch in {path}: "
        f"{len(raw_lines)} raw versus {len(hits)} parsed"
    )
    require(not hits.empty, f"Unexpected empty saved result: {path}")
    require(set(hits["grna"]) == {guide}, f"Unexpected query guide: {path}")
    require(hits["strand"].isin(["+", "-"]).all(), "Invalid hit strand.")
    require(hits["position"].ge(0).all(), "Invalid hit position.")
    require(hits["mismatches"].between(0, 4).all(), "Invalid mismatch count.")

    for row in hits.itertuples(index=False):
        sequence = row.offtarget_sequence
        require(
            len(sequence) == 20 + len(pam) and not (set(sequence) - set("ACGT")),
            "Unexpected hit sequence length or alphabet."
        )
        require(
            all(base in IUPAC[code] for base, code in zip(sequence[20:], pam)),
            "Hit does not satisfy the declared PAM rule."
        )
        actual_mm = sum(a != b for a, b in zip(guide, sequence[:20]))
        require(actual_mm == row.mismatches, "Mismatch count does not match sequence.")

    duplicate_mask = hits.duplicated(
        ["grna", "chrom", "position", "strand"], keep=False
    )
    require(
        not duplicate_mask.any(),
        "Duplicate genomic hit records found; investigate before ranking."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "results/release_validation/hfe_candidates_patched.csv",
    )
    parser.add_argument(
        "--saved-root",
        type=Path,
        default=ROOT / "results/hfe_offtarget_real",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results/release_validation/hfe_global_exact",
    )
    args = parser.parse_args()

    # Avoid silently overwriting a prior validation run.
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}. "
            "Choose another --outdir."
        )

    candidates = read_candidates(args.candidates)
    require(len(candidates) == 3, "Expected exactly three HFE candidates.")
    require(
        set(candidates["cas_variant"]) == set(RUNS),
        "Unexpected HFE variant set."
    )
    require(
        candidates["grna"].is_unique,
        "This validation requires unique guide sequences."
    )
    require(
        "bystander_weighted" in candidates,
        "Missing empirical bystander_weighted."
    )

    eff = pd.to_numeric(candidates["predicted_efficiency"], errors="raise")
    bystander = pd.to_numeric(candidates["bystander_weighted"], errors="raise")
    require(np.isfinite(eff).all() and eff.between(0, 1).all(),
            "Invalid efficiency predictions.")
    require(np.isfinite(bystander).all() and bystander.between(0, 1).all(),
            "Invalid bystander scores.")

    # Preserve old heuristic results for comparison, but do not use them
    # as the genome-wide ranking output.
    candidates = candidates.rename(columns={
        "composite_score": "heuristic_composite_score",
        "pareto_optimal": "heuristic_pareto_optimal",
    })

    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    run_metadata = []

    for variant, config in RUNS.items():
        sub = candidates[candidates["cas_variant"] == variant].copy()
        require(len(sub) == 1, f"Expected one candidate for {variant}.")
        guide = sub.iloc[0]["grna"]

        saved_dir = args.saved_root / config["directory"]
        input_path = saved_dir / "casoffinder_input.txt"
        raw_path = saved_dir / "casoffinder_output.txt"
        pam = config["pam"]

        genome_path_recorded = check_saved_input(input_path, guide, pam)
        hits = parse_casoffinder_output(raw_path)
        validate_parsed_hits(raw_path, hits, guide, pam)

        marked = mark_on_target(
            hits, sub, tolerance=0, pam_length=len(pam)
        )
        n_on = int(marked["is_on_target"].sum())
        require(
            n_on == 1,
            f"{variant}: expected one exact intended-target hit, found {n_on}."
        )

        summary = summarize(marked, max_mm=4)
        require(len(summary) == 1, f"Unexpected summary rows for {variant}.")
        require(
            int(summary.iloc[0]["total_offtargets"]) == len(hits) - n_on,
            "Hit accounting mismatch."
        )

        mismatch_cols = [f"hit_mm{i}" for i in range(5)]
        require(
            int(summary.iloc[0][mismatch_cols].sum())
            == int(summary.iloc[0]["total_offtargets"]),
            "Mismatch-bin total does not match off-target count."
        )

        variant_out = args.outdir / config["directory"]
        variant_out.mkdir()
        marked.to_csv(variant_out / "parsed_hits_exact.csv", index=False)
        summary.to_csv(variant_out / "summary.csv", index=False)
        summaries.append(summary)

        run_metadata.append({
            "variant": variant,
            "pam": pam,
            "max_mismatches": 4,
            "input_sha256": sha256(input_path),
            "raw_output_sha256": sha256(raw_path),
            "genome_path_recorded_in_original_input": genome_path_recorded,
            "genome_checksum_verified": False,
            "parsed_hits": len(hits),
            "intended_target_hits": n_on,
            "remaining_hits": int(summary.iloc[0]["total_offtargets"]),
        })

    combined = pd.concat(summaries, ignore_index=True)
    require(combined["grna"].is_unique, "Duplicate guides in combined summary.")
    require(
        set(combined["grna"]) == set(candidates["grna"]),
        "Candidate/summary guide sets differ."
    )
    combined.to_csv(args.outdir / "combined_summary.csv", index=False)

    # One normalization across all three HFE candidates.
    ranked = rerank(
        candidates, combined, scenario="therapy", w=(0.4, 0.3, 0.3)
    )

    raw = ranked["offtarget_risk_raw"].to_numpy(dtype=float)
    require(np.isfinite(raw).all() and (raw >= 0).all(), "Invalid raw risk.")
    expected_norm = (
        np.log1p(raw) / np.log1p(raw.max())
        if raw.max() > 0 else np.zeros_like(raw)
    )
    np.testing.assert_allclose(
        ranked["offtarget_risk_norm"], expected_norm
    )
    np.testing.assert_allclose(
        ranked["bystander_norm"], ranked["bystander_weighted"]
    )

    expected_score = (
        0.4 * ranked["predicted_efficiency"]
        - 0.3 * ranked["bystander_weighted"]
        - 0.3 * ranked["offtarget_risk_norm"]
    )
    np.testing.assert_allclose(
        ranked["composite_score_true_offtarget"], expected_score
    )

    # Keep numerical fields; rename historical "true" labels in the release output.
    ranked = ranked.rename(columns={
        "rank_true_offtarget": "rank_genome",
        "composite_score_true_offtarget": "composite_score_genome",
        "pareto": "pareto_genome",
    })
    ranked.to_csv(args.outdir / "candidates_ranked_global.csv", index=False)

    metadata = {
        "analysis": "HFE saved-output exact-coordinate reanalysis",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "command": sys.argv,
        "candidate_input_sha256": sha256(args.candidates),
        "parser_script_sha256": sha256(
            ROOT / "scripts/run_true_offtarget.py"
        ),
        "validation_script_sha256": sha256(Path(__file__)),
        "genome_search_rerun": False,
        "ranking_weights": {
            "efficiency": 0.4, "bystander": 0.3, "genome_score": 0.3
        },
        "risk_normalization": "log1p(raw)/log1p(max_raw), within this HFE candidate set",
        "bystander_metric": "empirical position-weighted score",
        "notes": [
            "Genome-hit weighting is heuristic, not an experimentally calibrated probability.",
            "Scores are candidate-set dependent and not directly comparable across loci.",
            "Saved input/output consistency was checked; genome-search completeness was not revalidated.",
            "Historical column names may remain in intermediate files.",
        ],
        "runs": run_metadata,
    }
    with (args.outdir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    columns = [
        "rank_genome", "grna", "cas_variant",
        "predicted_efficiency", "bystander_weighted",
        "total_offtargets", "offtarget_risk_raw",
        "offtarget_risk_norm", "composite_score_genome",
        "pareto_genome",
    ]
    print("\n=== HFE global ranking: exact-coordinate reanalysis ===")
    print(ranked[columns].to_string(index=False))
    print(f"\nSaved to: {args.outdir}")
    print("PASS: saved-output checks and global score calculations.")


if __name__ == "__main__":
    main()