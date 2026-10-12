"""
A/B comparison on the test set (novel-target proxy).

Plan A: full-feature model (0.751), 3 physical RE-COMPUTED, 24 energy MEAN-imputed.
Plan B: design model (0.685), seq(80) + 3 physical RE-COMPUTED, no energy.

Both use recomputed physical features (real sequence available in test set),
so this isolates the effect of "energy mean-imputed" (A) vs "energy dropped" (B).
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from compute_physical_features import compute_physical

TARGET = "efficiency_full_grna_reported"
ENERGY_COLS = [f"energy_{i}" for i in range(1, 25)]


def sequence_to_onehot(seq):
    mapping = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0],
               "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}
    out = []
    for n in seq.upper():
        out.extend(mapping.get(n, [0, 0, 0, 0]))
    return out


def evaluate(y, p, name):
    m = {
        "spearman": float(spearmanr(y, p)[0]),
        "pearson": float(pearsonr(y, p)[0]),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }
    print(f"{name:<28}{m['spearman']:>10.3f}{m['pearson']:>10.3f}"
          f"{m['mae']:>9.3f}{m['rmse']:>9.3f}")
    return m


def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    energy_mean = train[ENERGY_COLS].mean().values.astype(float)  # 24

    with open("models/xgboost_baseline.pkl", "rb") as f:
        model_a = pickle.load(f)   # 107-dim full-feature model (0.751)
    with open("models/xgboost_design.pkl", "rb") as f:
        model_b = pickle.load(f)   # 83-dim design model (0.685)

    y = test[TARGET].values

    # build both feature matrices with RECOMPUTED physical features
    Xa, Xb = [], []
    for _, r in test.iterrows():
        seq_feat = sequence_to_onehot(r["grna"])              # 80
        phys = compute_physical(r["grna"], r["sequence"])     # 3 recomputed
        # Plan A: 80 + 24 energy(mean) + 3 phys(recomputed)
        Xa.append(seq_feat + list(energy_mean) + phys)
        # Plan B: 80 + 3 phys(recomputed)
        Xb.append(seq_feat + phys)
    Xa = np.array(Xa, dtype=np.float32)
    Xb = np.array(Xb, dtype=np.float32)

    pa = np.clip(model_a.predict(Xa), 0, 1)
    pb = np.clip(model_b.predict(Xb), 0, 1)

    print("=" * 64)
    print("A/B on test set (energy MEAN-imputed for A, since novel-target proxy)")
    print("=" * 64)
    print(f"{'model':<28}{'Spearman':>10}{'Pearson':>10}{'MAE':>9}{'RMSE':>9}")
    print("-" * 64)
    evaluate(y, pa, "A: full-model, energy=mean")
    evaluate(y, pb, "B: design-model, no energy")
    print("=" * 64)
    print("\nReference: A with REAL energy (in-distribution) = Test Spearman 0.751")
    print("This table shows A/B when energy must be imputed (true novel-target case).")


if __name__ == "__main__":
    main()