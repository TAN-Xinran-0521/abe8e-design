"""
Design-specific model (Plan B): 80 one-hot + 3 RE-COMPUTED physical features.
NO energy_1..24 -> fully self-consistent recipe, zero mean-imputation for
novel targets. Recomputes physical features from sequence so train & inference
share one recipe. Compares against baseline (Test Spearman 0.751).
"""

import pandas as pd
import numpy as np
import pickle
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from compute_physical_features import compute_physical

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

TARGET = "efficiency_full_grna_reported"


def sequence_to_onehot(seq):
    mapping = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0],
               "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}
    out = []
    for n in seq.upper():
        out.extend(mapping.get(n, [0, 0, 0, 0]))
    return out


def build_X(df):
    rows = []
    for _, r in df.iterrows():
        seq_feat = sequence_to_onehot(r["grna"])           # 80
        phys = compute_physical(r["grna"], r["sequence"])  # 3 (recomputed)
        rows.append(seq_feat + phys)
    return np.array(rows, dtype=np.float32)


def evaluate(y_true, y_pred, name):
    m = {
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    print(f"{name:<8}{m['spearman']:>11.3f}{m['pearson']:>11.3f}"
          f"{m['mae']:>10.3f}{m['rmse']:>10.3f}")
    return m


def main():
    print("Loading splits and recomputing physical features (may take a bit)...")
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train, y_train = build_X(train), train[TARGET].values
    X_val, y_val = build_X(val), val[TARGET].values
    X_test, y_test = build_X(test), test[TARGET].values
    print(f"Feature dim: {X_train.shape[1]} (80 seq + 3 recomputed physical)\n")

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    with open("models/xgboost_design.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nSaved -> models/xgboost_design.pkl")

    print("\n" + "=" * 62)
    print("Plan B: seq(80) + recomputed physical(3), NO energy "
          "[baseline Test Spearman = 0.751]")
    print("=" * 62)
    print(f"{'Split':<8}{'Spearman':>11}{'Pearson':>11}{'MAE':>10}{'RMSE':>10}")
    print("-" * 62)
    res = {
        "train": evaluate(y_train, model.predict(X_train), "Train"),
        "val": evaluate(y_val, model.predict(X_val), "Val"),
        "test": evaluate(y_test, model.predict(X_test), "Test"),
    }
    print("=" * 62)

    import json
    with open("results/xgboost_design_results.json", "w") as f:
        json.dump({"model": "seq80+phys3_no_energy", "metrics": res}, f, indent=2)
    print("\nSaved results -> results/xgboost_design_results.json")


if __name__ == "__main__":
    main()