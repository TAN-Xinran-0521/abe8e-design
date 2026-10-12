"""Audit raw versus clipped B-prime predictions without retraining.

Loads only the trusted local research model. Pickle files must not be loaded
from untrusted sources.
"""
import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
import Bio
import RNA

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bprime_features import build_X, FEATURE_RECIPE_VERSION

ROOT = Path(__file__).resolve().parents[1]
TARGET = "efficiency_full_grna_reported"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, prediction):
    result = {
        "spearman": float(spearmanr(y, prediction)[0]),
        "pearson": float(pearsonr(y, prediction)[0]),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
    }
    require(
        all(np.isfinite(value) for value in result.values()),
        "Non-finite evaluation metric."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "models/xgboost_design_v2.pkl"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data/processed"
    )
    parser.add_argument(
        "--historical-results", type=Path,
        default=ROOT / "results/xgboost_design_v2_results.json"
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "results/release_validation/bprime_prediction_policy"
    )
    parser.add_argument(
        "--splits", nargs="+",
        choices=["train", "val", "test"],
        default=["val", "test"]
    )
    args = parser.parse_args()

    require(len(set(args.splits)) == len(args.splits), "Duplicate split names.")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}. "
            "Choose another --outdir."
        )

    with args.model.open("rb") as handle:
        model = pickle.load(handle)

    require(
        getattr(model, "n_features_in_", None) == 83,
        "Expected an 83-feature B-prime model."
    )

    with args.historical_results.open("r", encoding="utf-8-sig") as handle:
        historical = json.load(handle)

    args.outdir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    difference_rows = []
    summaries = {}
    input_checksums = {}

    for split in args.splits:
        source = args.data_dir / f"{split}.csv"
        frame = pd.read_csv(source)

        require(
            {"grna", TARGET}.issubset(frame.columns),
            f"{split}: missing required columns."
        )
        require(
            not frame[["grna", TARGET]].isna().any().any(),
            f"{split}: missing values; no rows will be silently removed."
        )

        y = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=float)
        require(len(y) > 1, f"{split}: too few samples.")
        require(
            np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all(),
            f"{split}: labels must be finite and within [0,1]."
        )

        print(f"Building shared B-prime features: {split}, n={len(frame)}")
        x = build_X(frame)
        require(
            x.shape == (len(frame), 83) and np.isfinite(x).all(),
            f"{split}: invalid feature matrix."
        )

        prediction_raw = np.asarray(model.predict(x), dtype=float)
        require(
            prediction_raw.shape == y.shape
            and np.isfinite(prediction_raw).all(),
            f"{split}: invalid predictions."
        )
        prediction_clipped = np.clip(prediction_raw, 0.0, 1.0)

        raw_metrics = metrics(y, prediction_raw)
        clipped_metrics = metrics(y, prediction_clipped)

        for policy, values in [
            ("raw", raw_metrics),
            ("clip_0_1", clipped_metrics)
        ]:
            metric_rows.append({
                "split": split,
                "prediction_policy": policy,
                "n": len(y),
                **values,
            })

        for metric, current in raw_metrics.items():
            previous = float(historical["metrics"][split][metric])
            difference_rows.append({
                "split": split,
                "metric": metric,
                "historical_raw": previous,
                "current_raw": current,
                "current_minus_historical": current - previous,
                "clipped_minus_raw": clipped_metrics[metric] - current,
            })

        below = int((prediction_raw < 0).sum())
        above = int((prediction_raw > 1).sum())
        summaries[split] = {
            "n": len(y),
            "raw_min": float(prediction_raw.min()),
            "raw_max": float(prediction_raw.max()),
            "n_below_zero": below,
            "n_above_one": above,
            "n_changed_by_clipping": below + above,
            "raw_metrics": raw_metrics,
            "clipped_metrics": clipped_metrics,
        }

        pd.DataFrame({
            "source_row_0based": np.arange(len(frame)),
            "grna": frame["grna"],
            "observed_efficiency": y,
            "prediction_raw": prediction_raw,
            "prediction_clipped": prediction_clipped,
        }).to_csv(args.outdir / f"{split}_predictions.csv", index=False)

        input_checksums[split] = sha256(source)

        print(
            f"{split}: below 0={below}, above 1={above}, "
            f"range=[{prediction_raw.min():.8f}, {prediction_raw.max():.8f}]"
        )

    metric_table = pd.DataFrame(metric_rows)
    differences = pd.DataFrame(difference_rows)

    metric_table.to_csv(args.outdir / "metrics_by_policy.csv", index=False)
    differences.to_csv(
        args.outdir / "historical_comparison.csv", index=False
    )

    manifest = {
        "purpose": "Read-only prediction-policy audit; not model selection.",
        "feature_recipe": FEATURE_RECIPE_VERSION,
        "model_sha256": sha256(args.model),
        "historical_results_sha256": sha256(args.historical_results),
        "split_sha256": input_checksums,
        "script_sha256": sha256(Path(__file__)),
        "feature_script_sha256": sha256(
            ROOT / "scripts/bprime_features.py"
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "biopython": Bio.__version__,
            "viennarna": getattr(RNA, "__version__", "unknown"),
        },
        "results": summaries,
        "notes": [
            "No retraining or data-split changes were performed.",
            "Historical metrics use raw predictions.",
            "Current design inference clips predictions to [0,1].",
            "Differences are reported rather than tuned away.",
            "This audit does not repair calibration-set reuse."
        ],
    }

    with (args.outdir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    print("\n=== Raw versus clipped metrics ===")
    print(metric_table.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    print("\n=== Historical comparison ===")
    print(differences.to_string(index=False, float_format=lambda x: f"{x:.12g}"))

    print(f"\nAudit completed: {args.outdir}")
    print("Completion means calculations succeeded, not that every historical value matched.")


if __name__ == "__main__":
    main()