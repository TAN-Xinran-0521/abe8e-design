"""
conformal_predict.py
Split conformal prediction for the B' design model (absolute-residual variant).

在 val 集上计算 B' 的绝对残差 |y - y_hat|，取 (1-alpha) 分位数（含有限样本
校正 ceil((n+1)(1-alpha))/n）作为固定区间半宽 q。test 集上区间为
[y_hat - q, y_hat + q]（在 [0,1] 截断），并报告测试集经验覆盖率；不保证有限测试样本的覆盖率达到名义水平。

特征口径与 B' / design_sgrna.py 逐字一致（80 one-hot + 3 grna 现算物理），
先做点估计闸门校验（B' test Spearman ~0.670），不符即中止（防口径漂移）。

产出：
  - conformal_qhat.json   两档 alpha 的半宽 q + 经验覆盖率 + 平均区间宽度
  - 供 design_sgrna.py 读取，给候选加 pred_eff_low / pred_eff_high
"""
import argparse, os, sys, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
import RNA
from Bio.SeqUtils import MeltingTemp as mt

TARGET = "efficiency_full_grna_reported"


from bprime_features import sequence_to_onehot, phys_from_grna, build_X

def predict(model, df):
    return np.clip(model.predict(build_X(df)), 0.0, 1.0)


def conformal_qhat(residuals, alpha):
    """有限样本校正的绝对残差分位数。"""
    residuals = np.asarray(residuals, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals must be non-empty and finite.")
    n = len(residuals)
    # 覆盖率保证需要的秩：ceil((n+1)(1-alpha))
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        raise ValueError("Requested level needs more calibration samples.")
    q = np.sort(residuals)[k - 1]
    return float(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=str(ROOT / "data/processed/val.csv"))
    ap.add_argument("--test", default=str(ROOT / "data/processed/test.csv"))
    ap.add_argument("--model", default=str(ROOT / "models/xgboost_design_v2.pkl"))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.1],
                    help="miscoverage levels; 0.2 -> 80%% PI, 0.1 -> 90%% PI")
    ap.add_argument("--outdir", default=str(ROOT / "results/conformal"))
    ap.add_argument("--exp_bprime", type=float, default=0.670)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    val = pd.read_csv(args.val)[["grna", TARGET]].dropna()
    val = val[val["grna"].str.len() == 20]
    test = pd.read_csv(args.test)[["grna", TARGET]].dropna()
    test = test[test["grna"].str.len() == 20]

    print("Predicting on val (calibration) + test...")
    yv, pv = val[TARGET].values.astype(float), predict(model, val)
    yt, pt = test[TARGET].values.astype(float), predict(model, test)

    # ---- 闸门校验：B' 点估计必须复现 ----
    s_bp = spearmanr(yt, pt)[0]
    print(f"\n[gate] B' test Spearman = {s_bp:.4f} (expect ~{args.exp_bprime})")
    if not np.isfinite(s_bp) or abs(s_bp - args.exp_bprime) > args.tol:
        print("[ABORT] 点估计未复现，特征口径可能漂移，中止。")
        sys.exit(1)
    print("[gate] passed.\n")

    resid_val = np.abs(yv - pv)     # 校准残差
    out = {"calibration_note": "Historical validation split was also used for early stopping; report empirical coverage only.", "model": "B'(seq80+grna_phys3)", "n_calib": int(len(resid_val)),
           "levels": {}}

    for alpha in args.alphas:
        q = conformal_qhat(resid_val, alpha)
        lo = np.clip(pt - q, 0.0, 1.0)
        hi = np.clip(pt + q, 0.0, 1.0)
        covered = np.mean((yt >= lo) & (yt <= hi))
        width = np.mean(hi - lo)
        pi = int(round((1 - alpha) * 100))
        print(f"=== {pi}% PI (alpha={alpha}) ===")
        print(f"  q_hat (half-width)     : {q:.4f}")
        print(f"  empirical coverage     : {covered:.4f}  (target >= {1-alpha:.2f})")
        print(f"  mean interval width    : {width:.4f}")
        print(f"  clipped-to-[0,1] note  : 若标签位于[0,1]，区间截断不改变样本覆盖指示，仅缩短区间\n")
        out["levels"][f"{pi}"] = {
            "alpha": alpha, "q_hat": q,
            "empirical_coverage": float(covered),
            "mean_width": float(width),
        }

    with open(f"{args.outdir}/conformal_qhat.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {args.outdir}/conformal_qhat.json")
    print("\n设计工具接入：读该 json 的 q_hat，对每条候选输出 "
          "pred_eff_low = clip(pred-q,0,1), pred_eff_high = clip(pred+q,0,1)")


if __name__ == "__main__":
    main()