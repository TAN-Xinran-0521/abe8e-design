"""
bootstrap_ci.py
Test 集 bootstrap 置信区间：对 baseline(107维) 与 B'(83维) 两个生产模型，
在同一 test.csv 上各用其原始特征配方 predict 一次，再对 (y_true, y_pred)
做 N 次有放回重采样，报告 Spearman/Pearson/MAE 的均值 + 95% CI。

口径零漂移保障：
  - baseline 特征逐字复用 train_baseline.py 的 extract_features
  - B' 特征逐字复用 train_design_model_v2.py 的 sequence_to_onehot + phys_from_grna
  - 先做点估计闸门校验：baseline≈0.751 / B'≈0.670，偏差过大即中止（防口径错位）
  - bootstrap 只重采样索引、重算指标，不重跑模型/特征
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import RNA
from Bio.SeqUtils import MeltingTemp as mt

TARGET = "efficiency_full_grna_reported"

# ---------- baseline (107-dim), 逐字复用 train_baseline.py ----------
def onehot_baseline(seq):
    mapping = {'A': [1,0,0,0], 'T': [0,1,0,0], 'G': [0,0,1,0], 'C': [0,0,0,1]}
    enc = []
    for n in seq:
        enc.extend(mapping.get(n, [0,0,0,0]))
    return enc

def features_baseline(df):
    grna = np.array(df["grna"].apply(onehot_baseline).tolist())
    energy = df[[f"energy_{i}" for i in range(1, 25)]].values
    phys = df[["free_energy", "melt_temperature_grna", "melt_temperature_target"]].values
    return np.hstack([grna, energy, phys])

# ---------- B' (83-dim), 逐字复用 train_design_model_v2.py ----------
def onehot_bprime(seq):
    mapping = {"A":[1,0,0,0], "T":[0,1,0,0], "G":[0,0,1,0], "C":[0,0,0,1]}
    out = []
    for n in seq.upper():
        out.extend(mapping.get(n, [0,0,0,0]))
    return out

def phys_from_grna(grna):
    g = grna.upper()
    rna = g.replace("T", "U")
    fc = RNA.fold_compound(rna)
    _, mfe = fc.mfe()
    tm_g = float(mt.Tm_NN(g))
    return [float(mfe), tm_g, tm_g]

def features_bprime(df):
    rows = [onehot_bprime(g) + phys_from_grna(g) for g in df["grna"]]
    return np.array(rows, dtype=np.float32)


def metrics(y, p):
    return {
        "spearman": float(spearmanr(y, p)[0]),
        "pearson": float(pearsonr(y, p)[0]),
        "mae": float(mean_absolute_error(y, p)),
    }

def bootstrap(y, p, n_boot, seed):
    rng = np.random.default_rng(seed)
    N = len(y)
    keys = ["spearman", "pearson", "mae"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)   # 有放回
        m = metrics(y[idx], p[idx])
        for k in keys:
            acc[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(acc[k])
        out[k] = (float(arr.mean()),
                  float(np.percentile(arr, 2.5)),
                  float(np.percentile(arr, 97.5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=__import__('repo_paths').repo_path('data/processed/test.csv'))
    ap.add_argument("--baseline", default=__import__('repo_paths').repo_path('models/xgboost_baseline.pkl'))
    ap.add_argument("--bprime", default=__import__('repo_paths').repo_path('models/xgboost_design_v2.pkl'))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=__import__('repo_paths').repo_path('results/bootstrap'))
    # 闸门校验期望值与容差
    ap.add_argument("--exp_baseline", type=float, default=0.751)
    ap.add_argument("--exp_bprime", type=float, default=0.670)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    test = pd.read_csv(args.test)
    y = test[TARGET].values.astype(float)

    with open(args.baseline, "rb") as f:
        m_base = pickle.load(f)
    with open(args.bprime, "rb") as f:
        m_bp = pickle.load(f)

    print("Computing features + predictions (once)...")
    Xb = features_baseline(test)
    pb = m_base.predict(Xb)
    Xp = features_bprime(test)      # 现算 ViennaRNA，约几十秒
    pp = m_bp.predict(Xp)

    # ---- 闸门校验：点估计必须复现 ----
    s_base = spearmanr(y, pb)[0]
    s_bp = spearmanr(y, pp)[0]
    print(f"\n[gate] baseline test Spearman = {s_base:.4f} (expect ~{args.exp_baseline})")
    print(f"[gate] B'       test Spearman = {s_bp:.4f} (expect ~{args.exp_bprime})")
    if abs(s_base - args.exp_baseline) > args.tol or abs(s_bp - args.exp_bprime) > args.tol:
        print("\n[ABORT] 点估计未复现，特征口径可能漂移，请检查配方后再跑 bootstrap。")
        sys.exit(1)
    print("[gate] passed. 口径自洽，开始 bootstrap...\n")

    rows = []
    for name, p in [("baseline(107)", pb), ("B'(83)", pp)]:
        ci = bootstrap(y, p, args.n_boot, args.seed)
        print(f"=== {name} (N_boot={args.n_boot}) ===")
        for k in ["spearman", "pearson", "mae"]:
            mean, lo, hi = ci[k]
            print(f"  {k:<9} {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print()
        for k in ["spearman", "pearson", "mae"]:
            mean, lo, hi = ci[k]
            rows.append({"model": name, "metric": k,
                         "mean": mean, "ci_low": lo, "ci_high": hi})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{args.outdir}/bootstrap_ci.csv", index=False)

    # ---- 图：Spearman 带 95% CI 误差棒 ----
    sp = df_out[df_out["metric"] == "spearman"].set_index("model")
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = range(len(sp))
    means = sp["mean"].values
    lows = means - sp["ci_low"].values
    highs = sp["ci_high"].values - means
    ax.bar(xs, means, yerr=[lows, highs], capsize=6,
           color=["#4C72B0", "#55A868"], alpha=0.85)
    ax.set_xticks(list(xs)); ax.set_xticklabels(sp.index)
    ax.set_ylabel("Test Spearman")
    ax.set_ylim(0.5, 0.85)
    for x, m in zip(xs, means):
        ax.text(x, m + 0.01, f"{m:.3f}", ha="center")
    ax.set_title(f"Test Spearman with 95% CI (bootstrap N={args.n_boot})")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/fig_bootstrap_spearman.png", dpi=150)

    print(f"Saved -> {args.outdir}/bootstrap_ci.csv, fig_bootstrap_spearman.png")


if __name__ == "__main__":
    main()