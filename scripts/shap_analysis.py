#!/usr/bin/env python3
"""SHAP analysis for design model B' (xgboost_design_v2.pkl).
Features: 80 one-hot (20nt x [A,T,G,C]) + 3 grna-only physical.
Uses xgboost NATIVE TreeSHAP (booster.predict(pred_contribs=True)) to avoid
the shap<->xgboost3.2 base_score parsing bug. Values are exact TreeSHAP.
Note: Tm_grna == Tm_target in B' (self-consistent); importance splits between them.
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import RNA
from Bio.SeqUtils import MeltingTemp as mt

ROOT = __import__('repo_paths').repo_path('')
MODEL = os.path.join(ROOT, "models/xgboost_design_v2.pkl")
TEST  = os.path.join(ROOT, "data/processed/test.csv")
OUT   = os.path.join(ROOT, "results/shap")
os.makedirs(OUT, exist_ok=True)

def sequence_to_onehot(seq):
    mapping = {"A":[1,0,0,0],"T":[0,1,0,0],"G":[0,0,1,0],"C":[0,0,0,1]}
    out = []
    for n in seq.upper():
        out.extend(mapping.get(n, [0,0,0,0]))
    return out

def phys_from_grna(grna):
    g = grna.upper(); rna = g.replace("T","U")
    _, mfe = RNA.fold_compound(rna).mfe()
    tm = float(mt.Tm_NN(g))
    return [float(mfe), tm, tm]

def build_X(df):
    return np.array([sequence_to_onehot(r["grna"]) + phys_from_grna(r["grna"])
                     for _, r in df.iterrows()], dtype=np.float32)

NUC = ["A","T","G","C"]
FEAT = [f"p{p}_{b}" for p in range(1,21) for b in NUC] + \
       ["free_energy","Tm_grna","Tm_target"]

def main():
    with open(MODEL,"rb") as f: model = pickle.load(f)
    booster = model.get_booster()
    test = pd.read_csv(TEST)
    X = build_X(test)
    print("X:", X.shape)

    # native xgboost TreeSHAP; last column is bias -> drop it
    dm = xgb.DMatrix(X)
    contribs = booster.predict(dm, pred_contribs=True, validate_features=False)
    sv = np.asarray(contribs)[:, :-1]   # (n, 83)
    print("SHAP values:", sv.shape)

    imp = np.abs(sv).mean(axis=0)
    pd.DataFrame({"feature":FEAT,"mean_abs_shap":imp}) \
      .to_csv(os.path.join(OUT,"feature_importance.csv"), index=False)

    pos_imp = [float(imp[p*4:p*4+4].sum()) for p in range(20)]
    pos_df = pd.DataFrame({"position":range(1,21),"importance":pos_imp})
    pos_df.to_csv(os.path.join(OUT,"position_importance.csv"), index=False)
    print("\nTop positions by importance:")
    print(pos_df.sort_values("importance",ascending=False).head(8).to_string(index=False))
    print("\nPhysical features:")
    print(pd.DataFrame({"feature":FEAT[-3:],"mean_abs_shap":imp[-3:]}).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9,4))
    bars = ax.bar(range(1,21), pos_imp, color="#4C72B0")
    for p in [4,5,6,7,8]: bars[p-1].set_color("#C44E52")
    ax.set_xlabel("protospacer position (1-20)")
    ax.set_ylabel("sum mean|SHAP| over A/T/G/C")
    ax.set_title("Per-position importance (edit window 4-8 in red)")
    ax.set_xticks(range(1,21))
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_position_importance.png"),dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.barh(FEAT[-3:], imp[-3:], color="#DD8452")
    ax.set_xlabel("mean|SHAP|"); ax.set_title("Physical feature importance")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_physical_importance.png"),dpi=150); plt.close(fig)

    shap.summary_plot(sv, X, feature_names=FEAT, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_beeswarm.png"),dpi=150); plt.close()
    print("\nOutputs ->", OUT)

if __name__ == "__main__":
    main()
