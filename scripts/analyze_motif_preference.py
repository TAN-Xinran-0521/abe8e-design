"""
analyze_motif_preference.py
复用 calibrate_bystander_prior.py 的单-A 归因口径，分析目标 A 的 5' 邻碱基 motif 偏好。
口径严格对齐：
  - 单-A：grna 在归因窗口 [attrib_lo, attrib_hi]=[1,12] 内恰好一个 A
  - grna 列 / efficiency_full_grna_reported / 长度=20，与 calibrate 脚本一致
新增：取该唯一 A 的 5' 邻碱基（protospacer position 1 = PAM-distal 为 5' 端，
      故 5' 邻碱基 = position p-1 = grna[p-2]），按碱基分组看 A->G 活性偏好。
产出两个口径：
  1) 原始分组（可能被位置混淆）
  2) 位置归一化分组（每条效率 / 其所在位置均值，剥离位置效应）——报告主用口径
  另加核心窗 4-8 内的稳健性复核。
与 Day4 SHAP、Day5 经验位置曲线形成第三条独立生物学印证线。
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASES = ["A", "C", "G", "T"]


def a_positions(grna, lo, hi):
    # 与 calibrate_bystander_prior.py 完全一致
    return [i for i in range(lo, hi + 1) if grna[i - 1] == 'A']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=__import__('repo_paths').repo_path('data/processed/train.csv'))
    ap.add_argument("--grna_col", default="grna")
    ap.add_argument("--eff_col", default="efficiency_full_grna_reported")
    ap.add_argument("--attrib_lo", type=int, default=1)
    ap.add_argument("--attrib_hi", type=int, default=12)
    ap.add_argument("--core_lo", type=int, default=4)
    ap.add_argument("--core_hi", type=int, default=8)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--outdir", default=__import__('repo_paths').repo_path('results/motif'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.train)[[args.grna_col, args.eff_col]].dropna()
    df = df[df[args.grna_col].str.len() == 20]

    # 单-A 归因 + 记录 5' 邻碱基
    rows = []
    for g, e in zip(df[args.grna_col], df[args.eff_col]):
        pos = a_positions(g, args.attrib_lo, args.attrib_hi)
        if len(pos) == 1:
            p = pos[0]
            if p < 2:                       # position 1 无 5' 邻碱基，跳过
                continue
            up = g[p - 2]                   # 5' 邻碱基 = position p-1
            rows.append((p, up, float(e)))
    sa = pd.DataFrame(rows, columns=["position", "up5", "efficiency"])
    print(f"可用单-A gRNA(含 5' 邻碱基, position>=2): {len(sa)} / {len(df)}")

    # 位置归一化：每条效率 / 其所在位置的均值（剥离位置效应）
    pos_mean = sa.groupby("position")["efficiency"].transform("mean")
    sa["eff_posnorm"] = sa["efficiency"] / pos_mean.replace(0, np.nan)

    # ---- 口径1：原始分组 ----
    raw = (sa.groupby("up5")["efficiency"]
             .agg(["count", "mean", "median", "std"])
             .reindex(BASES))
    print("\n[口径1] 原始 5' 邻碱基分组（未控位置）:")
    print(raw.to_string())

    # ---- 口径2：位置归一化分组（报告主用）----
    norm = (sa.dropna(subset=["eff_posnorm"])
              .groupby("up5")["eff_posnorm"]
              .agg(["count", "mean", "median", "std"])
              .reindex(BASES))
    print("\n[口径2] 位置归一化分组（剥离位置效应，报告主用）:")
    print(norm.to_string())

    # ---- 口径3：核心窗 4-8 内复核 ----
    core = sa[(sa["position"] >= args.core_lo) & (sa["position"] <= args.core_hi)]
    core_stat = (core.groupby("up5")["efficiency"]
                     .agg(["count", "mean", "median", "std"])
                     .reindex(BASES))
    print(f"\n[口径3] 核心窗 {args.core_lo}-{args.core_hi} 内分组（稳健性复核）:")
    print(core_stat.to_string())

    # motif × position 交叉表（归一化均值），看偏好是否跨位置一致
    pivot = (sa.dropna(subset=["eff_posnorm"])
               .pivot_table(index="up5", columns="position",
                            values="eff_posnorm", aggfunc="mean")
               .reindex(BASES))

    # 保存
    raw.to_csv(f"{args.outdir}/motif_raw.csv")
    norm.to_csv(f"{args.outdir}/motif_posnorm.csv")
    core_stat.to_csv(f"{args.outdir}/motif_core_window.csv")
    pivot.to_csv(f"{args.outdir}/motif_by_position.csv")

    # 偏好排序（以位置归一化均值为准）
    order = norm["mean"].dropna().sort_values(ascending=False)
    ranking = " > ".join(f"{b}A" for b in order.index)
    print(f"\n5' 邻碱基偏好排序（位置归一化均值）: {ranking}")

    # ---- 图：三口径 XA motif 偏好 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (tab, col, title) in zip(
        axes,
        [(raw, "mean", "Raw (position-confounded)"),
         (norm, "mean", "Position-normalized (main)"),
         (core_stat, "mean", f"Core window {args.core_lo}-{args.core_hi}")]):
        vals = tab["mean"].reindex(BASES)
        errs = tab["std"].reindex(BASES)
        ax.bar([f"{b}A" for b in BASES], vals.values,
               yerr=errs.values, capsize=4, color="#4C72B0", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("5' neighbor + target A")
        ax.set_ylabel("A->G activity" if col == "mean" else "")
    plt.suptitle("ABE8e 5'-neighbor motif preference (single-A attribution)")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/fig_motif_preference.png", dpi=150)

    # ---- 图：motif × position 热图 ----
    plt.figure(figsize=(8, 3))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="position-normalized activity")
    plt.yticks(range(len(BASES)), [f"{b}A" for b in BASES])
    plt.xticks(range(pivot.shape[1]), pivot.columns)
    plt.xlabel("protospacer position of target A (1 = PAM-distal)")
    plt.title("Motif preference across positions")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/fig_motif_by_position.png", dpi=150)

    print(f"\n已保存 -> {args.outdir}/ (motif_*.csv, fig_motif_preference.png, fig_motif_by_position.png)")


if __name__ == "__main__":
    main()