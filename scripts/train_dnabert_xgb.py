"""
train_dnabert_xgb.py
用 DNABERT-2 embedding(768维) 训 XGBoost 效率回归，对标 B'(one-hot 80+3物理, Test Spearman 0.670)。
公平对照：XGBoost 超参与 B' 完全一致，唯一变量是特征编码器。
额外做一个 late-fusion 变体：768 embedding + 3 grna 现算物理(与B'同配方)，看能否叠加增益。
"""
import os, json, pickle
os.environ.setdefault("HF_HOME", "/mnt/d/TZS/hf_cache")

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

FEAT = "/mnt/d/TZS/data/features_dnabert"
OUTDIR = "/mnt/d/TZS/results/dnabert"
MODELDIR = "/mnt/d/TZS/models"


def load(split):
    d = np.load(f"{FEAT}/{split}.npz")
    return d["X"], d["y"]


def evaluate(y, p, name):
    m = {
        "spearman": float(spearmanr(y, p)[0]),
        "pearson": float(pearsonr(y, p)[0]),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }
    print(f"{name:<10}{m['spearman']:>10.3f}{m['pearson']:>10.3f}"
          f"{m['mae']:>9.3f}{m['rmse']:>9.3f}")
    return m


def train_xgb(Xtr, ytr, Xval, yval):
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def run(tag, Xtr, ytr, Xval, yval, Xte, yte, results):
    print(f"\n=== {tag} (dim={Xtr.shape[1]}) ===")
    print(f"{'Split':<10}{'Spearman':>10}{'Pearson':>10}{'MAE':>9}{'RMSE':>9}")
    model = train_xgb(Xtr, ytr, Xval, yval)
    res = {
        "train": evaluate(ytr, model.predict(Xtr), "Train"),
        "val": evaluate(yval, model.predict(Xval), "Val"),
        "test": evaluate(yte, model.predict(Xte), "Test"),
    }
    results[tag] = {"dim": int(Xtr.shape[1]), "metrics": res}
    return model


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(MODELDIR, exist_ok=True)

    Xtr, ytr = load("train")
    Xval, yval = load("val")
    Xte, yte = load("test")
    print(f"Loaded DNABERT embeddings: train{Xtr.shape} val{Xval.shape} test{Xte.shape}")

    results = {}

    # --- 变体1：纯 DNABERT-2 embedding (768) ---
    m1 = run("dnabert768", Xtr, ytr, Xval, yval, Xte, yte, results)
    with open(f"{MODELDIR}/xgboost_dnabert768.pkl", "wb") as f:
        pickle.dump(m1, f)

    # --- 变体2：late-fusion, 768 + 3 grna现算物理(与B'同配方) ---
    # 复用 B' 的 phys_from_grna 保证口径一致
    import RNA
    from Bio.SeqUtils import MeltingTemp as mt
    import pandas as pd

    def phys_from_grna(g):
        g = g.upper(); rna = g.replace("T", "U")
        _, mfe = RNA.fold_compound(rna).mfe()
        tmg = float(mt.Tm_NN(g))
        return [float(mfe), tmg, tmg]

    def add_phys(split, X):
        df = pd.read_csv(f"/mnt/d/TZS/data/processed/{split}.csv")
        df = df[df["efficiency_full_grna_reported"].notna()]
        df = df[df["grna"].str.len() == 20].reset_index(drop=True)
        phys = np.array([phys_from_grna(g) for g in df["grna"]], dtype=np.float32)
        assert len(phys) == len(X), f"{split} 行数不匹配: phys{len(phys)} vs X{len(X)}"
        return np.hstack([X, phys])

    print("\nBuilding late-fusion features (768 + 3 physical)...")
    Xtr2 = add_phys("train", Xtr)
    Xval2 = add_phys("val", Xval)
    Xte2 = add_phys("test", Xte)
    m2 = run("dnabert768+phys3", Xtr2, ytr, Xval2, yval, Xte2, yte, results)
    with open(f"{MODELDIR}/xgboost_dnabert_fusion.pkl", "wb") as f:
        pickle.dump(m2, f)

    # --- 汇总对标 ---
    print("\n" + "=" * 60)
    print("对标 (Test Spearman):")
    print(f"  B' (one-hot 80 + phys 3)      : 0.670   [基线]")
    print(f"  Transformer 纯序列 (80)        : 0.668")
    print(f"  DNABERT-2 (768)               : {results['dnabert768']['metrics']['test']['spearman']:.3f}")
    print(f"  DNABERT-2 + phys3 (771)       : {results['dnabert768+phys3']['metrics']['test']['spearman']:.3f}")
    print(f"  全特征 baseline (含 energy)     : 0.751   [信息上限]")
    print("=" * 60)

    with open(f"{OUTDIR}/dnabert_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTDIR}/dnabert_results.json, models/xgboost_dnabert*.pkl")


if __name__ == "__main__":
    main()