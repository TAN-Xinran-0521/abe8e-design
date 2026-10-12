"""
visualize_candidates.py
把 ΔΔG + 三方法共识结果映射到 TadA-8e 结构上。
输入:
  - D:/TZS/data/structures/tada8e_structure.pdb   (ESMFold 预测, 残基1..166)
  - D:/TZS/results/ddg_consistency/priority_quadrant.csv
输出 (D:/TZS/results/figures/):
  - tada8e_candidates.html      py3Dmol 交互图: 功能区着色 + 44候选位点标记 + C145热点
  - tada8e_c145_zoom.html       C145 三方法共识热点特写
  - fig_ddg_structure_2d.png    matplotlib 2D摘要(残基位置 vs ΔΔG, 不依赖浏览器)
说明: HTML 用浏览器打开可旋转并导出高清PNG(右键或截图); 2D PNG 直接可进PPT。
口径: 结构为ESMFold预测(非实验); ΔΔG为力场估计, 取定性排序。
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol

# 功能区 -> 残基范围(序列原生编号, 与PDB一致), 用于结构着色
REGION_RANGES = {
    "n_terminal":       (list(range(40, 55)),  "#8172B3"),
    "catalytic_core":   (list(range(55, 100)), "#C44E52"),
    "substrate_channel":(list(range(100, 135)),"#4C72B0"),
    "specificity_loop": (list(range(140, 160)),"#55A868"),
}
QUAD_STYLE = {   # quadrant -> (marker color, 说明)
    "both_support": "#00A000",   # 两方支持(绿)
    "ddg_only":     "#4C72B0",   # 仅稳定化(蓝)
    "lm_only":      "#DDAA00",   # 仅语言模型(黄)
    "neither":      "#999999",   # 皆不(灰)
}


def read_pdb(path):
    with open(path) as f:
        return f.read()


def add_region_coloring(view):
    """按功能区给 cartoon 上色, 其余灰。"""
    view.setStyle({"cartoon": {"color": "#DDDDDD"}})
    for reg, (resis, color) in REGION_RANGES.items():
        view.setStyle({"resi": [str(r) for r in resis]},
                      {"cartoon": {"color": color}})


def mark_candidates(view, df):
    """把候选位点画成球, 颜色按 quadrant; C145 用大球+标签。"""
    for _, r in df.iterrows():
        pos = int(r["mutation"][1:-1])
        color = QUAD_STYLE.get(r["quadrant"], "#999999")
        is_c145 = pos == 145
        view.addStyle({"resi": str(pos)},
                      {"sphere": {"color": color,
                                  "radius": 1.6 if is_c145 else 0.9}})
    # C145 标签(三方法共识热点)
    view.addLabel("C145 hotspot (3-method consensus)",
                  {"fontColor": "white", "backgroundColor": "#C44E52",
                   "fontSize": 12},
                  {"resi": "145"})


def build_html(pdb_str, df, out_html, zoom_resi=None):
    view = py3Dmol.view(width=900, height=700)
    view.addModel(pdb_str, "pdb")
    add_region_coloring(view)
    mark_candidates(view, df)
    if zoom_resi is not None:
        view.zoomTo({"resi": str(zoom_resi)})
    else:
        view.zoomTo()
    view.setBackgroundColor("white")
    html = view._make_html()
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def plot_2d_summary(df, out_png):
    """matplotlib 2D: 残基位置(x) vs ΔΔG(y), 按region着色, 标共识位点。不依赖浏览器。"""
    plt.figure(figsize=(11, 5))
    reg_color = {reg: c for reg, (_, c) in REGION_RANGES.items()}
    for reg, sub in df.groupby("region"):
        pos = sub["mutation"].str[1:-1].astype(int)
        plt.scatter(pos, sub["ddg_foldx"],
                    c=reg_color.get(reg, "gray"), label=reg, s=55,
                    edgecolors="k", linewidths=0.4)
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    # 标注 C145 系列 + 显著去稳定化点
    for _, r in df.iterrows():
        pos = int(r["mutation"][1:-1])
        if pos == 145 or r["ddg_foldx"] > 1.0 or r["ddg_foldx"] < -1.0:
            plt.annotate(r["mutation"], (pos, r["ddg_foldx"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("TadA-8e residue position (native numbering, 1-166)")
    plt.ylabel("FoldX ΔΔG (kcal/mol; <0 stabilizing)")
    plt.title("Candidate mutations across TadA-8e: position vs stability")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(out_png, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=__import__('repo_paths').repo_path('data/structures/tada8e_structure.pdb'))
    ap.add_argument("--priority",
                    default=__import__('repo_paths').repo_path('results/ddg_consistency/priority_quadrant.csv'))
    ap.add_argument("--outdir", default=__import__('repo_paths').repo_path('results/figures'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pdb_str = read_pdb(args.pdb)
    df = pd.read_csv(args.priority)
    print(f"候选数: {len(df)} | quadrant 分布:")
    print(df["quadrant"].value_counts().to_string())

    html_full = os.path.join(args.outdir, "tada8e_candidates.html")
    html_zoom = os.path.join(args.outdir, "tada8e_c145_zoom.html")
    png_2d = os.path.join(args.outdir, "fig_ddg_structure_2d.png")

    build_html(pdb_str, df, html_full, zoom_resi=None)
    build_html(pdb_str, df, html_zoom, zoom_resi=145)
    plot_2d_summary(df, png_2d)

    print(f"\nSaved:")
    print(f"  {html_full}   <- 浏览器打开, 旋转后导出高清PNG(PPT全局图)")
    print(f"  {html_zoom}   <- C145 三方法共识热点特写")
    print(f"  {png_2d}      <- 2D摘要, 直接可进PPT(不依赖浏览器)")
    print("\n图例: 绿=both_support 蓝=ddg_only 黄=lm_only 灰=neither; "
          "红大球+标签=C145热点")
    print("功能区着色: 紫=N端 红=催化核心 蓝=底物通道 绿=特异性环")


if __name__ == "__main__":
    main()