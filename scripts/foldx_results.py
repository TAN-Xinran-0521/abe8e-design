"""Strict parser for the inspected FoldX BuildModel output convention.

Supported: single substitutions, one-character chain IDs, explicit model prefix,
Average/Dif/Raw outputs with tab-delimited headers.
No FoldX process is executed.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_mutation(text, sequence):
    match = re.fullmatch(r"([A-Z])([1-9][0-9]*)([A-Z])", str(text))
    require(match is not None, f"Invalid substitution: {text}")
    wt, position, alt = match.group(1), int(match.group(2)), match.group(3)
    require(wt in AA and alt in AA and wt != alt, f"Invalid amino acids: {text}")
    require(1 <= position <= len(sequence), f"Position outside reference: {text}")
    require(sequence[position - 1] == wt, f"Reference residue mismatch: {text}")
    return wt, position, alt


def read_mutations(path, sequence, chain):
    require(len(chain) == 1 and chain.isalnum(), "Expected one-character chain ID.")
    mutations = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        match = re.fullmatch(
            r"([A-Z])([A-Za-z0-9])([1-9][0-9]*)([A-Z]);", line.strip()
        )
        require(match is not None, f"Unsupported mutation line {number}: {line}")
        require(match.group(2) == chain, f"Chain mismatch at line {number}.")
        mutation = match.group(1) + match.group(3) + match.group(4)
        parse_mutation(mutation, sequence)
        mutations.append(mutation)
    require(mutations and len(mutations) == len(set(mutations)),
            "Mutation list is empty or contains duplicates.")
    return mutations


def read_energy_table(path, with_sd=False):
    path = Path(path)
    header = None
    records = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = [value.strip() for value in line.split("\t")]
        if header is None:
            if parts[0] == "Pdb" and "total energy" in parts:
                header = parts
                needed = ["Pdb", "total energy"] + (["SD"] if with_sd else [])
                require(
                    all(header.count(name) == 1 for name in needed),
                    f"Missing or duplicate required header: {path}"
                )
            continue
        indexes = [header.index("Pdb"), header.index("total energy")]
        if with_sd:
            indexes.append(header.index("SD"))
        require(len(parts) > max(indexes), f"Short data row {number}: {path}")
        try:
            energy = float(parts[header.index("total energy")])
            sd = float(parts[header.index("SD")]) if with_sd else None
        except ValueError as exc:
            raise ValueError(f"Non-numeric data row {number}: {path}") from exc
        require(np.isfinite(energy), f"Non-finite energy at row {number}: {path}")
        if with_sd:
            require(np.isfinite(sd) and sd >= 0, f"Invalid SD at row {number}: {path}")
        records.append({
            "pdb": parts[header.index("Pdb")],
            "energy": energy,
            "sd": sd,
        })

    require(header is not None and records, f"No energy table found: {path}")
    frame = pd.DataFrame(records)
    require(frame["pdb"].is_unique, f"Duplicate model identifier: {path}")
    return frame.set_index("pdb")


def build_results(workdir, prefix, sequence, candidates_path, chain="A", runs=5):
    workdir = Path(workdir)
    require(isinstance(runs, int) and runs >= 1, "runs must be positive.")
    require(prefix and "/" not in prefix and "\\" not in prefix,
            "prefix must be a file-name prefix, not a path.")
    sequence = sequence.upper()
    require(sequence and not (set(sequence) - AA), "Invalid reference sequence.")

    mutations = read_mutations(
        workdir / "individual_list.txt", sequence, chain
    )
    candidates = pd.read_csv(candidates_path)
    require("mutation" in candidates, "Candidate table lacks mutation column.")
    require(candidates["mutation"].is_unique, "Duplicate candidate mutations.")
    require(set(candidates["mutation"]) == set(mutations),
            "Candidate table and mutation list must contain exactly the same substitutions.")
    candidates = candidates.set_index("mutation")

    average = read_energy_table(workdir / f"Average_{prefix}.fxout", with_sd=True)
    dif = read_energy_table(workdir / f"Dif_{prefix}.fxout")
    raw = read_energy_table(workdir / f"Raw_{prefix}.fxout")

    expected_average = {f"{prefix}_{i}" for i in range(1, len(mutations) + 1)}
    expected_dif = {
        f"{prefix}_{i}_{j}.pdb"
        for i in range(1, len(mutations) + 1) for j in range(runs)
    }
    require(set(average.index) == expected_average, "Average model identifiers differ.")
    require(set(dif.index) == expected_dif, "Dif model identifiers/repeat count differ.")
    require(
        set(raw.index) == expected_dif | {"WT_" + name for name in expected_dif},
        "Raw mutant/WT identifiers differ."
    )

    result_rows, repeat_rows = [], []
    raw_errors, mean_errors, sd_errors = [], [], []

    for index, mutation in enumerate(mutations, 1):
        values = []
        for repeat in range(runs):
            identifier = f"{prefix}_{index}_{repeat}.pdb"
            difference = float(dif.loc[identifier, "energy"])
            mutant_energy = float(raw.loc[identifier, "energy"])
            wt_energy = float(raw.loc["WT_" + identifier, "energy"])
            error = mutant_energy - wt_energy - difference
            raw_errors.append(abs(error))
            values.append(difference)
            repeat_rows.append({
                "mutation": mutation,
                "model_id": identifier,
                "repeat": repeat,
                "mutant_energy": mutant_energy,
                "matched_wt_energy": wt_energy,
                "ddg_dif": difference,
                "raw_minus_wt_vs_dif_error": error,
            })

        avg = average.loc[f"{prefix}_{index}"]
        mean = float(avg["energy"])
        sd = float(avg["sd"])
        mean_errors.append(abs(mean - float(np.mean(values))))
        sd_errors.append(abs(sd - float(np.std(values, ddof=0))))

        wt, position, alt = parse_mutation(mutation, sequence)
        row = {
            "mutation": mutation,
            "model_index": index,
            "chain": chain,
            "reference_position": position,
            "reference_residue": wt,
            "alternate_residue": alt,
            "ddg_foldx": mean,
            "ddg_sd": sd,
            "ddg_mean_from_dif": float(np.mean(values)),
            "ddg_sd_population_from_dif": float(np.std(values, ddof=0)),
            "n_runs": runs,
            "stability": (
                "stabilizing" if mean < 0
                else "destabilizing" if mean > 0
                else "zero"
            ),
        }
        metadata = candidates.loc[mutation]
        for column in ["region", "n_models", "models"]:
            if column in candidates.columns:
                row[column] = metadata[column]
        if "mean_delta" in candidates.columns:
            row["lm_mean_delta"] = float(metadata["mean_delta"])
        result_rows.append(row)

    diagnostics = {
        "n_mutations": len(mutations),
        "runs": runs,
        "max_raw_difference_error": max(raw_errors),
        "max_average_mean_error": max(mean_errors),
        "max_population_sd_error": max(sd_errors),
        "sd_convention": "population SD across computational repeats (ddof=0)",
        "tolerances": {
            "raw_difference": 0.001,
            "average_mean": 0.0001,
            "population_sd": 0.0001,
        },
    }

    require(max(raw_errors) <= 0.001, "Raw mutant-WT and Dif disagree beyond text precision tolerance.")
    require(max(mean_errors) <= 0.0001, "Average and mean Dif disagree.")
    require(max(sd_errors) <= 0.0001, "Reported SD differs from validated population-SD convention.")

    results = pd.DataFrame(result_rows).sort_values(
        ["ddg_foldx", "mutation"], kind="stable"
    ).reset_index(drop=True)
    diagnostics["predicted_stabilizing_count"] = int((results["ddg_foldx"] < 0).sum())
    return results, pd.DataFrame(repeat_rows), diagnostics