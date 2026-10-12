"""
diagnose_dnabert.py
诊断 DNABERT-2 embedding 的 Test Spearman=0.343 是"embedding差"还是"用法过拟合"。
三个对照(均在现有768维embedding上, 不需RNA):
  1. Ridge 线性探针(标准embedding评估, 看线性可解码信息)
  2. PCA降维(50/100维) + XGBoost
  3. 强正则 XGBoost(小depth+大min_child_weight+L1/L2)
对标: B' 0.670 / 纯序列Transformer 0.668
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb

FEAT = "/mnt/d/TZS/data/features_dnabert"


def load(s):
    d = np.load(f"{FEAT}/{s}.npz"); return d["X"], d["y"]


def sp(y, p): return spearmanr(y, p)[0]


def main():
    Xtr, ytr = load("train"); Xval, yval = load("val"); Xte, yte = load("test")
    print(f"embeddings: train{Xtr.shape} test{Xte.shape}\n")

    # ---- 1. Ridge 线性探针(标准化后) ----
    sc = StandardScaler().fit(Xtr)
    Ztr, Zval, Zte = sc.transform(Xtr), sc.transform(Xval), sc.transform(Xte)
    print("=== Ridge linear probe (embedding 质量标准评估) ===")
    for alpha in [1.0, 10.0, 100.0]:
        r = Ridge(alpha=alpha).fit(Ztr, ytr)
        print(f"  alpha={alpha:<6} Train {sp(ytr, r.predict(Ztr)):.3f}  "
              f"Val {sp(yval, r.predict(Zval)):.3f}  Test {sp(yte, r.predict(Zte)):.3f}")

    # ---- 2. PCA + XGBoost ----
    print("\n=== PCA 降维 + XGBoost (与B'同超参) ===")
    for k in [50, 100]:
        pca = PCA(n_components=k, random_state=42).fit(Ztr)
        Ptr, Pval, Pte = pca.transform(Ztr), pca.transform(Zval), pca.transform(Zte)
        ev = pca.explained_variance_ratio_.sum()
        m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, early_stopping_rounds=50)
        m.fit(Ptr, ytr, eval_set=[(Pval, yval)], verbose=False)
        print(f"  PCA-{k} (var{ev:.2f})  Train {sp(ytr, m.predict(Ptr)):.3f}  "
              f"Val {sp(yval, m.predict(Pval)):.3f}  Test {sp(yte, m.predict(Pte)):.3f}")

    # ---- 3. 强正则 XGBoost(原768维) ----
    print("\n=== 强正则 XGBoost (768维, 抗过拟合) ===")
    m = xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.03,
                         subsample=0.7, colsample_bytree=0.3,
                         min_child_weight=10, reg_alpha=1.0, reg_lambda=5.0,
                         random_state=42, early_stopping_rounds=50)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    print(f"  Train {sp(ytr, m.predict(Xtr)):.3f}  "
          f"Val {sp(yval, m.predict(Xval)):.3f}  Test {sp(yte, m.predict(Xte)):.3f}")

    print("\n对标: B' 0.670 | 纯序列Transformer 0.668 | baseline 0.751")
    print("判读: 若 Ridge/PCA/正则后 Test 仍<<0.67 -> embedding本身对该任务信息不足;")
    print("      若某法回到~0.6+ -> 原0.343是XGBoost在768维过拟合(用法问题)。")


if __name__ == "__main__":
    main()