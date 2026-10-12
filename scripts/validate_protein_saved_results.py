"""Read-only audit of saved FoldX outputs and protein-model score matrices.

No FoldX execution, no model inference, no historical-output overwrite.
Supports the inspected single-chain, single-substitution file format.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
COL = {a: i for i, a in enumerate(AA)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_energy_table(path, require_sd=False):
    """Use literal tab-delimited header names; retain Pdb identifiers."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = None
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t")]
        if header is None:
            if parts[0] == "Pdb" and "total energy" in parts:
                header = parts
                require(
                    header.count("total energy") == 1,
                    f"Ambiguous energy header: {path}"
                )
                if require_sd:
                    require(header.count("SD") == 1, f"Missing SD: {path}")
            continue

        index_energy = header.index("total energy")
        needed = [0, index_energy]
        if require_sd:
            needed.append(header.index("SD"))
        require(
            len(parts) > max(needed),
            f"Malformed data row {line_number}: {path}"
        )
        try:
            energy = float(parts[index_energy])
            sd = float(parts[header.index("SD")]) if require_sd else None
        except ValueError as exc:
            raise ValueError(f"Non-numeric row {line_number}: {path}") from exc
        require(np.isfinite(energy), f"Non-finite energy: {path}")
        if require_sd:
            require(np.isfinite(sd) and sd >= 0, f"Invalid SD: {path}")
        rows.append({"pdb": parts[0], "energy": energy, "sd": sd})

    require(header is not None and rows, f"No valid table: {path}")
    frame = pd.DataFrame(rows)
    require(frame["pdb"].is_unique, f"Duplicate Pdb identifiers: {path}")
    return frame.set_index("pdb")


def parse_mutation(mutation, sequence):
    match = re.fullmatch(r"([A-Z])([1-9][0-9]*)([A-Z])", str(mutation))
    require(match is not None, f"Invalid mutation: {mutation}")
    wt, pos, alt = match.group(1), int(match.group(2)), match.group(3)
    require(wt in COL and alt in COL and wt != alt, f"Invalid substitution: {mutation}")
    require(1 <= pos <= len(sequence), f"Position out of range: {mutation}")
    require(sequence[pos - 1] == wt, f"WT mismatch: {mutation}")
    return wt, pos, alt


def delta(matrix, mutation, sequence):
    wt, pos, alt = parse_mutation(mutation, sequence)
    # Retain the saved matrix's subtraction semantics, without decimal rounding.
    return float(matrix[pos - 1, COL[alt]] - matrix[pos - 1, COL[wt]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "results/release_validation/protein_saved_audit")
    args = ap.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose another --outdir.")

    fxdir = ROOT / "results/foldx"
    prefix = "tada8e_structure_Repair"
    files = {
        "individual": fxdir / "individual_list.txt",
        "average": fxdir / f"Average_{prefix}.fxout",
        "dif": fxdir / f"Dif_{prefix}.fxout",
        "raw": fxdir / f"Raw_{prefix}.fxout",
        "candidates": ROOT / "results/deaminase_compare/candidate_consensus_exact.csv",
        "historical_ddg": fxdir / "ddg_results.csv",
        "fasta": ROOT / "data/structures/tada8e_sequence.fasta",
    }

    sequences = list(SeqIO.parse(str(files["fasta"]), "fasta"))
    require(len(sequences) == 1, "Expected one reference FASTA record.")
    sequence = str(sequences[0].seq).upper()
    require(len(sequence) == 166 and not (set(sequence) - set(AA)),
            "Unexpected reference sequence.")

    mutations = []
    for line in files["individual"].read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([A-Z])([A-Za-z0-9])([1-9][0-9]*)([A-Z]);", line.strip())
        require(match is not None, f"Unsupported FoldX mutation line: {line}")
        require(match.group(2) == "A", "This audit expects chain A.")
        mutation = match.group(1) + match.group(3) + match.group(4)
        parse_mutation(mutation, sequence)
        mutations.append(mutation)

    require(len(mutations) == 44 and len(set(mutations)) == 44,
            "Expected 44 unique substitutions.")

    candidates = pd.read_csv(files["candidates"])
    historical = pd.read_csv(files["historical_ddg"])
    for name, frame in [("candidate", candidates), ("historical", historical)]:
        require(frame["mutation"].is_unique, f"Duplicate {name} mutations.")
        require(set(frame["mutation"]) == set(mutations),
                f"{name} mutation set differs from individual_list.")
    candidates = candidates.set_index("mutation")
    historical = historical.set_index("mutation")

    average = read_energy_table(files["average"], require_sd=True)
    differences = read_energy_table(files["dif"])
    raw = read_energy_table(files["raw"])

    expected_average = {f"{prefix}_{i}" for i in range(1, 45)}
    expected_dif = {f"{prefix}_{i}_{j}.pdb" for i in range(1, 45) for j in range(5)}
    expected_raw = expected_dif | {"WT_" + name for name in expected_dif}

    require(set(average.index) == expected_average, "Unexpected Average model IDs.")
    require(set(differences.index) == expected_dif, "Unexpected Dif model IDs/repeats.")
    require(set(raw.index) == expected_raw, "Unexpected Raw mutant/WT model IDs.")

    rows = []
    repeat_rows = []
    for index, mutation in enumerate(mutations, start=1):
        vals = []
        for repeat in range(5):
            identifier = f"{prefix}_{index}_{repeat}.pdb"
            mutant_energy = float(raw.loc[identifier, "energy"])
            wt_energy = float(raw.loc["WT_" + identifier, "energy"])
            difference = float(differences.loc[identifier, "energy"])
            error = (mutant_energy - wt_energy) - difference
            vals.append(difference)
            repeat_rows.append({
                "mutation": mutation, "model_id": identifier,
                "mutant_energy": mutant_energy, "matched_wt_energy": wt_energy,
                "difference_from_raw": mutant_energy - wt_energy,
                "difference_reported": difference,
                "raw_difference_error": error,
            })

        av = average.loc[f"{prefix}_{index}"]
        reported_mean = float(av["energy"])
        reported_sd = float(av["sd"])
        old_mean = float(historical.loc[mutation, "ddg_foldx"])
        old_sd = float(historical.loc[mutation, "ddg_sd"])

        rows.append({
            "mutation": mutation,
            "model_index": index,
            "region": candidates.loc[mutation, "region"],
            "n_models": int(candidates.loc[mutation, "n_models"]),
            "ddg_average": reported_mean,
            "ddg_mean_from_dif": float(np.mean(vals)),
            "mean_error": reported_mean - float(np.mean(vals)),
            "sd_reported": reported_sd,
            "sd_population": float(np.std(vals, ddof=0)),
            "sd_sample": float(np.std(vals, ddof=1)),
            "historical_ddg": old_mean,
            "historical_sd": old_sd,
            "historical_ddg_error": old_mean - reported_mean,
            "historical_sd_error": old_sd - reported_sd,
            "ddg_foldx": reported_mean,
        })

    audit = pd.DataFrame(rows)
    repeats = pd.DataFrame(repeat_rows)

    # Numerical-file text precision is limited; retain errors for inspection.
    raw_error = float(repeats["raw_difference_error"].abs().max())
    mean_error = float(audit["mean_error"].abs().max())
    old_error = float(audit["historical_ddg_error"].abs().max())
    old_sd_error = float(audit["historical_sd_error"].abs().max())

    model_specs = [
        ("esm2_35m", "deaminase"),
        ("esm1v_650m", "deaminase_650m"),
        ("esm_if1", "deaminase_if1"),
    ]
    gate_records = []
    for name, directory in model_specs:
        matrix_path = ROOT / "results" / directory / "logprob_matrix.npy"
        table_path = ROOT / "results" / directory / "candidate_mutations.csv"
        files[name + "_matrix"] = matrix_path
        files[name + "_candidates"] = table_path

        matrix = np.load(matrix_path, allow_pickle=False)
        require(matrix.shape == (166, 20) and np.isfinite(matrix).all(),
                f"Invalid matrix: {name}")
        table = pd.read_csv(table_path)
        require(len(table) > 0 and table["mutation"].is_unique,
                f"Empty/duplicate candidate table: {name}")

        errors = []
        for record in table.itertuples(index=False):
            value = float(record.delta_logp)
            require(np.isfinite(value), f"Non-finite candidate score: {name}")
            errors.append(abs(delta(matrix, record.mutation, sequence) - value))

        maximum = float(max(errors))
        require(maximum <= 0.01, f"{name}: matrix/CSV score disagreement {maximum}")
        gate_records.append({"model": name, "n_checked": len(table),
                             "max_absolute_error": maximum, "tolerance": 0.01})
        audit[name] = [delta(matrix, m, sequence) for m in mutations]

    audit["esm_seq_mean"] = (audit["esm2_35m"] + audit["esm1v_650m"]) / 2

    correlation_rows = []
    for policy in ["full_precision", "historical_3decimal"]:
        y = (audit["ddg_average"] if policy == "full_precision"
             else audit["historical_ddg"])
        for name in ["esm2_35m", "esm1v_650m", "esm_seq_mean", "esm_if1"]:
            x = audit[name] if policy == "full_precision" else audit[name].map(
                lambda value: round(float(value), 3)
            )
            rho, p = spearmanr(x, y)
            require(np.isfinite(rho) and np.isfinite(p), "Undefined correlation.")
            correlation_rows.append({
                "policy": policy, "model_score": name, "n": len(audit),
                "spearman": float(rho), "p_value": float(p),
            })

    correlations = pd.DataFrame(correlation_rows)
    report = {
        "n_mutations": len(audit),
        "repeats_per_mutation": 5,
        "predicted_stabilizing_average_lt_zero": int((audit["ddg_average"] < 0).sum()),
        "maximum_raw_minus_wt_vs_dif_error": raw_error,
        "maximum_average_vs_mean_dif_error": mean_error,
        "maximum_historical_ddg_rounding_error": old_error,
        "maximum_historical_sd_rounding_error": old_sd_error,
        "max_reported_sd_vs_population_sd_error": float(
            (audit["sd_reported"] - audit["sd_population"]).abs().max()
        ),
        "max_reported_sd_vs_sample_sd_error": float(
            (audit["sd_reported"] - audit["sd_sample"]).abs().max()
        ),
        "checks": {
            "raw_dif_within_0.001": raw_error <= 0.001,
            "average_mean_within_0.0001": mean_error <= 0.0001,
            "historical_ddg_within_0.000501": old_error <= 0.000501,
            "historical_sd_within_0.000501": old_sd_error <= 0.000501,
        },
        "matrix_checks": gate_records,
        "sha256": {name: checksum(path) for name, path in files.items()},
        "limitations": [
            "Reanalysis of saved outputs, not a new FoldX run.",
            "SD reflects repeated computational runs, not experimental uncertainty.",
            "Sequence-score average is an exploratory uncalibrated aggregation.",
            "Correlation is conditional on this selected candidate set.",
            "Structural methods share predicted-structure input.",
            "Tolerance checks account for printed precision; inspect failures, do not tune them away.",
        ],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.outdir / "mutation_audit.csv", index=False)
    repeats.to_csv(args.outdir / "repeat_energy_audit.csv", index=False)
    correlations.to_csv(args.outdir / "correlations.csv", index=False)
    with (args.outdir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("\n=== FoldX audit ===")
    for key, value in report.items():
        if key not in ["sha256", "limitations", "matrix_checks"]:
            print(f"{key}: {value}")
    print("\n=== Matrix/CSV checks ===")
    print(pd.DataFrame(gate_records).to_string(index=False))
    print("\n=== Correlations ===")
    print(correlations.to_string(index=False))
    print("\n=== Representative substitutions ===")
    selected = audit["mutation"].str.startswith("C145") | audit["mutation"].isin(
        ["S108P", "F83L", "P151R"]
    )
    print(audit.loc[selected, [
        "mutation", "n_models", "ddg_average", "sd_reported"
    ]].to_string(index=False))
    print(f"\nSaved audit to: {args.outdir}")

    if not all(report["checks"].values()):
        raise RuntimeError("One or more numerical checks failed. Inspect saved audit files.")
    print("PASS: structural file checks, matrix checks and numerical consistency checks.")


if __name__ == "__main__":
    main()