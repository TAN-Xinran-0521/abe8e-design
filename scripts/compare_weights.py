"""
compare_weights.py
读已有候选表(pred_efficiency / bystander_weighted / offtarget_risk 三分量已存好),
在多套权重下重算 composite 并排序,对比排序稳定性。不重跑模型。
composite = w1*eff - w2*bystander_weighted - w3*offtarget_risk
"""
import argparse
import pandas as pd

WEIGHT_SETS = {
    "current (0.4/0.3/0.3)":  (0.4, 0.3, 0.3),
    "proposed(0.5/0.2/0.3)":  (0.5, 0.2, 0.3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=__import__('repo_paths').repo_path('results/hfe_candidates_bystander.csv'))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    eff = df["pred_efficiency"]
    bys = df["bystander_weighted"]
    ot  = df["offtarget_risk"]

    for name, (w1, w2, w3) in WEIGHT_SETS.items():
        comp = (w1 * eff - w2 * bys - w3 * ot).round(3)
        order = comp.rank(ascending=False, method="min").astype(int)
        out = pd.DataFrame({
            "grna": df["grna"],
            "cas_variant": df["cas_variant"],
            "eff": eff,
            "bys_w": bys.round(3),
            "ot": ot,
            "composite": comp,
            "rank": order,
        }).sort_values("rank").reset_index(drop=True)
        print("\n" + "=" * 70)
        print(f"weights = {name}")
        print("=" * 70)
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()