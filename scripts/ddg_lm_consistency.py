"""Compare saved protein-model scores with validated FoldX results.

CONSISTENCY_RELEASE_V2
No model inference or FoldX execution.
Full precision is used for statistics; plots are optional.
No uncalibrated cross-model priority score is produced.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from Bio import SeqIO
from scipy.stats import spearmanr

from foldx_results import parse_mutation, require
from repo_paths import repo_path

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delta_from_matrix(matrix, mutation, sequence):
    wt, position, alt = parse_mutation(mutation, sequence)
    # Preserve the saved array's arithmetic; do not decimal-round scores.
    return float(
        matrix[position - 1, AA_INDEX[alt]]
        - matrix[position - 1, AA_INDEX[wt]]
    )


def load_model_scores(matrix_path, candidate_path, sequence, tolerance):
    matrix = np.load(matrix_path, allow_pickle=False)
    require(
        matrix.shape == (len(sequence), 20)
        and np.issubdtype(matrix.dtype, np.number)
        and np.isfinite(matrix).all(),
        f"Invalid score matrix: {matrix_path}"
    )

    candidates = pd.read_csv(candidate_path)
    require(
        {"mutation", "delta_logp"}.issubset(candidates.columns),
        f"Missing candidate columns: {candidate_path}"
    )
    require(
        len(candidates) > 0 and candidates["mutation"].is_unique,
        f"Empty or duplicate candidate table: {candidate_path}"
    )

    errors = []
    for row in candidates.itertuples(index=False):
        observed = float(row.delta_logp)
        require(np.isfinite(observed), "Non-finite candidate-table score.")
        calculated = delta_from_matrix(matrix, row.mutation, sequence)
        errors.append(abs(calculated - observed))

    maximum = float(max(errors))
    require(
        maximum <= tolerance,
        f"Matrix/CSV disagreement: {candidate_path}; max error={maximum}"
    )
    return matrix, set(candidates["mutation"]), {
        "n_checked": len(candidates),
        "maximum_absolute_error": maximum,
        "absolute_tolerance": tolerance,
    }


def make_plots(frame, correlations, outdir, plt):
    colors = {
        "catalytic_core": "#C44E52",
        "substrate_channel": "#4C72B0",
        "specificity_loop": "#55A868",
        "n_terminal": "#8172B3",
    }
    for score, title in [
        ("esm_if1", "ESM-IF1 structure-conditioned score"),
        ("esm_seq_mean", "Exploratory mean of two sequence-model scores"),
    ]:
        statistic = correlations.set_index("model_score").loc[score]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for region, subset in frame.groupby("region", dropna=False):
            ax.scatter(
                subset[score], subset["ddg_foldx"],
                label=str(region), color=colors.get(region, "gray"),
                edgecolors="black", linewidths=0.35, s=40,
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        for row in frame.itertuples(index=False):
            if row.mutation.startswith("C145") or row.mutation in ["S108P", "P151R"]:
                ax.annotate(
                    row.mutation, (getattr(row, score), row.ddg_foldx),
                    xytext=(3, 3), textcoords="offset points", fontsize=7,
                )
        ax.set_xlabel(title + "\nHigher = greater model preference")
        ax.set_ylabel("Predicted FoldX mean DDG (kcal/mol)")
        ax.set_title(
            f"n={len(frame)}, Spearman={statistic['spearman']:.3f}, "
            f"p={statistic['p_value']:.4g}"
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_ddg_vs_{score}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foldx", type=Path, required=True,
        help="Full-precision ddg_results.csv from the validated shared parser."
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--fasta", type=Path,
        default=Path(repo_path("data/structures/tada8e_sequence.fasta"))
    )
    parser.add_argument("--score-tolerance", type=float, default=0.01)
    parser.add_argument("--plots", action="store_true",
                        help="Generate optional plots; requires matplotlib.")

    specs = [
        ("esm2_35m", "m35", "c35", "deaminase"),
        ("esm1v_650m", "m650", "c650", "deaminase_650m"),
        ("esm_if1", "mif1", "cif1", "deaminase_if1"),
    ]
    for name, matrix_arg, candidate_arg, folder in specs:
        parser.add_argument(
            "--" + matrix_arg, type=Path,
            default=Path(repo_path(f"results/{folder}/logprob_matrix.npy"))
        )
        parser.add_argument(
            "--" + candidate_arg, type=Path,
            default=Path(repo_path(f"results/{folder}/candidate_mutations.csv"))
        )
    args = parser.parse_args()

    require(
        np.isfinite(args.score_tolerance) and args.score_tolerance >= 0,
        "Score tolerance must be finite and nonnegative."
    )
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose another --outdir.")

    plt = None
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    records = list(SeqIO.parse(str(args.fasta), "fasta"))
    require(len(records) == 1, "Expected exactly one reference FASTA record.")
    sequence = str(records[0].seq).upper()
    require(len(sequence) == 166 and not (set(sequence) - set(AA_ORDER)),
            "Expected the 166-residue reference sequence.")

    frame = pd.read_csv(args.foldx)
    require(
        {"mutation", "ddg_foldx", "ddg_sd", "n_runs"}.issubset(frame.columns),
        "Expected full-precision output from the revised FoldX parser."
    )
    require(len(frame) >= 3 and frame["mutation"].is_unique,
            "Too few or duplicate FoldX mutations.")
    for mutation in frame["mutation"]:
        parse_mutation(mutation, sequence)
    for column in ["ddg_foldx", "ddg_sd", "n_runs"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        require(np.isfinite(frame[column]).all(), f"Non-finite {column}.")
    require(frame["ddg_sd"].ge(0).all(), "Negative SD.")
    require(
        frame["n_runs"].ge(1).all()
        and (frame["n_runs"] % 1 == 0).all(),
        "Invalid number of computational runs."
    )
    if "region" not in frame:
        frame["region"] = "unannotated"

    files = {"foldx": args.foldx, "fasta": args.fasta}
    gate_records = []
    nomination_columns = []

    for name, matrix_arg, candidate_arg, folder in specs:
        matrix_path = getattr(args, matrix_arg)
        candidate_path = getattr(args, candidate_arg)
        matrix, nominations, gate = load_model_scores(
            matrix_path, candidate_path, sequence, args.score_tolerance
        )
        frame[name] = [
            delta_from_matrix(matrix, mutation, sequence)
            for mutation in frame["mutation"]
        ]
        nominated = name + "_nominated"
        frame[nominated] = frame["mutation"].isin(nominations)
        nomination_columns.append(nominated)
        gate_records.append({"model": name, **gate})
        files[name + "_matrix"] = matrix_path
        files[name + "_candidates"] = candidate_path

    frame["nomination_count_checked"] = frame[nomination_columns].sum(axis=1)
    if "n_models" in frame:
        historical_count = pd.to_numeric(frame["n_models"], errors="raise")
        require(
            np.array_equal(historical_count, frame["nomination_count_checked"]),
            "Candidate nomination counts disagree with the three source tables."
        )

    frame["esm_seq_mean"] = (frame["esm2_35m"] + frame["esm1v_650m"]) / 2
    frame["negative_mean_ddg"] = frame["ddg_foldx"] < 0

    correlation_rows = []
    for score in ["esm2_35m", "esm1v_650m", "esm_seq_mean", "esm_if1"]:
        require(
            frame[score].nunique() > 1 and frame["ddg_foldx"].nunique() > 1,
            f"Correlation undefined for constant values: {score}"
        )
        rho, p = spearmanr(frame[score], frame["ddg_foldx"])
        require(np.isfinite(rho) and np.isfinite(p), "Undefined correlation.")
        correlation_rows.append({
            "model_score": score,
            "n": len(frame),
            "spearman": float(rho),
            "p_value": float(p),
            "p_value_policy": "two-sided, unadjusted",
            "score_policy": "full precision; saved matrix subtraction",
        })
    correlations = pd.DataFrame(correlation_rows)

    manifest = {
        "analysis_version": "consistency_release_v2",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "n_candidates": len(frame),
        "negative_mean_ddg_count": int(frame["negative_mean_ddg"].sum()),
        "matrix_column_order": AA_ORDER,
        "matrix_candidate_checks": gate_records,
        "input_sha256": {name: sha256(path) for name, path in files.items()},
        "script_sha256": sha256(Path(__file__)),
        "mutation_validation_module_sha256": sha256(
            Path(__file__).with_name("foldx_results.py")
        ),
        "plots_generated": bool(args.plots),
        "limitations": [
            "Saved-output analysis; no FoldX or ESM execution.",
            "Protein-model scores are not calibrated functional-effect estimates.",
            "Sequence-score mean is an exploratory aggregation.",
            "P-values are unadjusted across the four reported correlations.",
            "Results are conditional on this selected candidate set.",
            "Structural methods share predicted-structure input.",
            "Nomination count means presence in source candidate tables, not proven function.",
            "No median-based consensus label or cross-model priority score is produced.",
            "No automatic literature-numbering conversion is performed.",
        ],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "lm_ddg_merged.csv", index=False)
    correlations.to_csv(args.outdir / "correlations.csv", index=False)
    pd.DataFrame(gate_records).to_csv(
        args.outdir / "matrix_validation.csv", index=False
    )
    if args.plots:
        make_plots(frame, correlations, args.outdir, plt)
    with (args.outdir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    print("\n=== Matrix/candidate checks ===")
    print(pd.DataFrame(gate_records).to_string(index=False))
    print("\n=== Full-precision correlations ===")
    print(correlations.to_string(index=False))
    print(
        f"\nMean DDG < 0: {manifest['negative_mean_ddg_count']}/{len(frame)}"
    )
    print(f"Saved to: {args.outdir}")


if __name__ == "__main__":
    main()