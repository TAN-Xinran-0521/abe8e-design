"""Reparse saved FoldX outputs using the validated shared parser.

Requires an explicit new output directory. Does not execute FoldX.
"""
import argparse
import hashlib
import json
from pathlib import Path

from Bio import SeqIO

from foldx_results import build_results, require
from repo_paths import repo_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path,
                        default=Path(repo_path("results/foldx")))
    parser.add_argument("--candidates", type=Path, default=Path(repo_path(
        "results/deaminase_compare/candidate_consensus_exact.csv"
    )))
    parser.add_argument("--fasta", type=Path,
                        default=Path(repo_path("data/structures/tada8e_sequence.fasta")))
    parser.add_argument("--prefix", default="tada8e_structure_Repair")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    source = args.workdir.resolve()
    destination = args.outdir.resolve()
    require(destination != source, "Output must not be the input work directory.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Output directory is not empty; choose another --outdir.")

    records = list(SeqIO.parse(str(args.fasta), "fasta"))
    require(len(records) == 1, "Expected one FASTA reference sequence.")
    sequence = str(records[0].seq).upper()

    results, repeats, diagnostics = build_results(
        source, args.prefix, sequence, args.candidates,
        chain=args.chain, runs=args.runs
    )

    inputs = {
        "individual_list": source / "individual_list.txt",
        "average": source / f"Average_{args.prefix}.fxout",
        "dif": source / f"Dif_{args.prefix}.fxout",
        "raw": source / f"Raw_{args.prefix}.fxout",
        "candidates": args.candidates,
        "fasta": args.fasta,
        "parser_module": Path(__file__).with_name("foldx_results.py"),
        "entry_script": Path(__file__),
    }
    diagnostics["input_sha256"] = {
        name: sha256(path) for name, path in inputs.items()
    }
    diagnostics["foldx_executed"] = False
    diagnostics["numbering"] = "Input reference numbering; no automatic literature +1 conversion."
    diagnostics["limitations"] = [
        "Saved-output reanalysis only.",
        "SD describes computational-repeat variation, not experimental or total model uncertainty.",
        "Negative mean energy difference is a predicted stability tendency, not functional validation.",
        "The actual PDB/reference mapping must be verified before a new FoldX computation.",
    ]

    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "ddg_results.csv", index=False)
    repeats.to_csv(destination / "repeat_energy_checks.csv", index=False)
    with (destination / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)

    print(results[["mutation", "ddg_foldx", "ddg_sd", "stability"]].to_string(index=False))
    print(
        f"\nPredicted mean DDG < 0: "
        f"{diagnostics['predicted_stabilizing_count']}/{diagnostics['n_mutations']}"
    )
    print("SD convention: population SD of computational repeats.")
    print(f"Saved to: {destination}")


if __name__ == "__main__":
    main()