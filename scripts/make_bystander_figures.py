"""
make_bystander_figures.py
本次 bystander 升级的三张报告图，全部读现成文件，不重跑模型。
  图B: SHAP 位置重要性 × 经验编辑窗活性曲线（双重印证）
  图C: HFE 三候选 bystander 计数 vs 位置加权 对比
  图D: 权重敏感性（两套权重下 composite 与排序）
输出 -> results/figures/
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"   # 全英文标签,避免中文字体缺失


def fig_b(root, outdir):
    """图B: SHAP 位置重要性 vs 经验活性曲线,各自归一化到峰=1 叠加。"""
    prior = json.load(open(f"{root}/results/bystander/empirical_prior.json"))
    act = pd.Series({int(k): float(v) for k, v in prior.items()}).sort_index()

    shap = pd.read_csv(f"{root}/results/shap/position_importance.csv")
    # 容错：找 position 列与 importance 列
    pcol = "position" if "position" in shap.columns else shap.columns[0]
    icol = "importance" if "importance" in shap.columns else shap.columns[1]
    s = shap.set_index(pcol)[icol]
    s = s.reindex(range(1, 21))
    s_norm = s / s.max()

    x = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x, act.reindex(range(1, 21)).values, width=0.55, color="#4C72B0",
           alpha=0.85, label="Empirical activity (single-A attribution)")
    ax.plot(x, s_norm.values, "-o", color="#C44E52", lw=2, ms=5,
            label="SHAP position importance (B' model)")
    ax.axvspan(3.5, 8.5, color="orange", alpha=0.12, label="edit window 4-8")
    ax.set_xlabel("protospacer position (1 = PAM-distal)")
    ax.set_ylabel("normalized value (peak = 1)")
    ax.set_title("Editing-window prior: SHAP × empirical activity (independent agreement)")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = f"{outdir}/figB_shap_vs_empirical.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


def fig_c(root, outdir):
    """图C: HFE 三候选 bystander_A(计数) vs bystander_weighted(加权)。"""
    df = pd.read_csv(f"{root}/results/hfe_candidates_bystander.csv")
    labels = [f"{v}\n({g[:8]}..)" for v, g in zip(df["cas_variant"], df["grna"])]
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w/2, df["bystander_A"], w, color="#8C8C8C",
           label="old: bystander_A (count, window 4-8)")
    ax.bar(x + w/2, df["bystander_weighted"], w, color="#4C72B0",
           label="new: bystander_weighted (position-weighted)")
    for xi, (c, wt) in enumerate(zip(df["bystander_A"], df["bystander_weighted"])):
        ax.text(xi - w/2, c + 0.02, str(int(c)), ha="center", fontsize=8)
        ax.text(xi + w/2, wt + 0.02, f"{wt:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("bystander risk")
    ax.set_title("HFE candidates: bystander count vs position-weighted\n"
                 "(rank1 SpRY: count=0 but a weak pos-9 bystander caught by weighting)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = f"{outdir}/figC_bystander_count_vs_weighted.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


def fig_d(root, outdir):
    """图D: 两套权重下 composite 与排序（读现成三分量重算,不跑模型）。"""
    df = pd.read_csv(f"{root}/results/hfe_candidates_bystander.csv")
    eff, bys, ot = df["pred_efficiency"], df["bystander_weighted"], df["offtarget_risk"]
    sets = {"current (0.4/0.3/0.3)": (0.4, 0.3, 0.3),
            "proposed (0.5/0.2/0.3)": (0.5, 0.2, 0.3)}
    labels = [f"{v}\n({g[:8]}..)" for v, g in zip(df["cas_variant"], df["grna"])]
    x = np.arange(len(df))
    w = 0.38
    colors = ["#4C72B0", "#DD8452"]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, (name, (w1, w2, w3)) in enumerate(sets.items()):
        comp = (w1 * eff - w2 * bys - w3 * ot)
        rank = comp.rank(ascending=False, method="min").astype(int)
        off = (i - 0.5) * w
        ax.bar(x + off, comp, w, color=colors[i], label=name)
        for xi, (cv, rk) in enumerate(zip(comp, rank)):
            ax.text(xi + off, cv + (0.004 if cv >= 0 else -0.012),
                    f"#{rk}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("composite score")
    ax.set_title("Weight sensitivity: ranking stable across weight sets\n"
                 "(SpRY > SpCas9 > SpCas9-NG under both)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = f"{outdir}/figD_weight_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=__import__('repo_paths').repo_path(''))
    args = ap.parse_args()
    outdir = f"{args.root}/results/figures"
    os.makedirs(outdir, exist_ok=True)
    fig_b(args.root, outdir)
    fig_c(args.root, outdir)
    fig_d(args.root, outdir)
    print("\nAll figures saved to", outdir)


if __name__ == "__main__":
    main()