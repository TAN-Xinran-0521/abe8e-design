"""
Design model B' (self-consistent recipe): all 3 physical features computed
from the 20nt protospacer (gRNA) ALONE, so training and design-tool inference
use the IDENTICAL input. No energy, no mean-imputation, no coordinate gap.

  free_energy             = MFE(grna as RNA)      [ViennaRNA]
  melt_temperature_grna   = Tm_NN(grna)           [Biopython]
  melt_temperature_target = Tm_NN(grna)           [same 20nt; self-consistent]

Rationale: design candidates scanned from arbitrary context only provide the
protospacer, not BE-dataHIVE's 40bp `sequence` column. Diagnostics showed
grna != sequence[10:30] (Pearson 0.879), so reusing the DB target-Tm recipe
would create a train/inference mismatch. B' eliminates that gap.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import RNA
from Bio.SeqUtils import MeltingTemp as mt




TARGET = "efficiency_full_grna_reported"


from bprime_features import sequence_to_onehot, phys_from_grna, build_X

def evaluate(y, p, name):
    m = {
        "spearman": float(spearmanr(y, p)[0]),
        "pearson": float(pearsonr(y, p)[0]),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }
    print(f"{name:<8}{m['spearman']:>11.3f}{m['pearson']:>11.3f}"
          f"{m['mae']:>10.3f}{m['rmse']:>10.3f}")
    return m


def main():
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    print("Loading splits, computing B' physical features from grna...")
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    val = pd.read_csv(ROOT / "data/processed/val.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")

    X_train, y_train = build_X(train), train[TARGET].values
    X_val, y_val = build_X(val), val[TARGET].values
    X_test, y_test = build_X(test), test[TARGET].values
    print(f"Feature dim: {X_train.shape[1]} (80 seq + 3 grna-only physical)\n")

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    with open(ROOT / "models/xgboost_design_v2.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nSaved -> models/xgboost_design_v2.pkl")

    print("\n" + "=" * 62)
    print("Model B': seq(80) + grna-only physical(3), self-consistent recipe")
    print("[baseline 0.751 | B(seq+phys w/ DB target Tm) 0.685]")
    print("=" * 62)
    print(f"{'Split':<8}{'Spearman':>11}{'Pearson':>11}{'MAE':>10}{'RMSE':>10}")
    print("-" * 62)
    res = {
        "train": evaluate(y_train, model.predict(X_train), "Train"),
        "val": evaluate(y_val, model.predict(X_val), "Val"),
        "test": evaluate(y_test, model.predict(X_test), "Test"),
    }
    print("=" * 62)

    with open(ROOT / "results/xgboost_design_v2_results.json", "w", encoding="utf-8") as f:
        json.dump({"model": "seq80+grna_phys3_selfconsistent",
                   "metrics": res}, f, indent=2)
    print("\nSaved results -> results/xgboost_design_v2_results.json")


if __name__ == "__main__":
    main()