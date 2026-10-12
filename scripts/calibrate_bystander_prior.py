"""
calibrate_bystander_prior.py
用本项目训练集反推 ABE8e 经验位置活性曲线(单-A 归因)：
若某 gRNA 在可编辑区内只有一个 A(位于 p 位), 则其 reported 效率归因于 p 位活性。
按 p 分箱取均值 -> 归一化(峰=1.0) -> 存为 bystander 先验。
把 bystander 项从文献先验升级为数据自洽(同 B' 哲学)。
"""
import argparse, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def a_positions(grna, lo, hi):
    return [i for i in range(lo, hi + 1) if grna[i - 1] == 'A']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/processed/train.csv"))
    ap.add_argument("--grna_col", default="grna")
    ap.add_argument("--eff_col", default="efficiency_full_grna_reported")
    ap.add_argument("--attrib_lo", type=int, default=1)
    ap.add_argument("--attrib_hi", type=int, default=12)
    ap.add_argument("--min_count", type=int, default=20,
                    help="每个位置可信估计所需的最少 gRNA 数")
    ap.add_argument("--outdir", default=str(ROOT / "results/bystander"))
    args = ap.parse_args()
    if not 1 <= args.attrib_lo <= args.attrib_hi <= 20:
        ap.error("Require 1 <= attrib_lo <= attrib_hi <= 20.")
    if args.min_count < 1:
        ap.error("min_count must be positive.")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.train)[[args.grna_col, args.eff_col]].dropna()
    df[args.grna_col] = df[args.grna_col].str.upper().str.strip()
    if not df[args.grna_col].str.fullmatch("[ACGT]{20}").all():
        raise ValueError("Training guides must be 20nt A/C/G/T strings.")
    df[args.eff_col] = pd.to_numeric(df[args.eff_col], errors="raise")
    if not np.isfinite(df[args.eff_col]).all() or not df[args.eff_col].between(0, 1).all():
        raise ValueError("Efficiency labels must be finite and within [0,1].")

    rows = []
    for g, e in zip(df[args.grna_col], df[args.eff_col]):
        pos = a_positions(g, args.attrib_lo, args.attrib_hi)
        if len(pos) == 1:                       # 单-A 归因
            rows.append((pos[0], float(e)))
    sa = pd.DataFrame(rows, columns=["position", "efficiency"])
    print(f"可用单-A gRNA: {len(sa)} / {len(df)}")

    stat = (sa.groupby("position")["efficiency"]
              .agg(["count", "mean", "median", "std"])
              .reindex(range(args.attrib_lo, args.attrib_hi + 1)))
    print("\n各位置单-A 活性:")
    print(stat.to_string())

    trusted = stat[stat["count"] >= args.min_count]
    if trusted.empty:
        print("\n[WARN] 无位置达到 --min_count；放宽 attrib 窗口或降低 min_count，"
              "否则建议回退文献先验。")
        return

    peak = trusted["mean"].max()
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Cannot normalize a non-positive or non-finite peak.")
    curve = pd.Series(index=range(1, 21), dtype=float)
    for p in range(1, 21):
        if p in trusted.index:
            curve[p] = trusted.loc[p, "mean"] / peak
    curve = curve.interpolate(limit_direction="both")   # 内部空缺线性插值+两端补齐
    curve = curve.clip(lower=0.01).round(4)
    prior = {int(p): float(v) for p, v in curve.items()}

    with open(f"{args.outdir}/empirical_prior.json", "w") as f:
        json.dump(prior, f, indent=2)
    stat.to_csv(f"{args.outdir}/position_activity.csv")

    provenance = []
    trusted_positions = list(trusted.index)
    for p in range(1, 21):
        if p in trusted.index:
            source = "observed"
        elif min(trusted_positions) < p < max(trusted_positions):
            source = "interpolated"
        else:
            source = "boundary_filled"
        if prior[p] <= 0.01:
            source += ";floor_at_0.01"
        provenance.append({
            "position": p,
            "sample_count": stat.loc[p, "count"] if p in stat.index else np.nan,
            "observed_mean": stat.loc[p, "mean"] if p in stat.index else np.nan,
            "final_weight": prior[p],
            "weight_source": source,
        })
    pd.DataFrame(provenance).to_csv(
        f"{args.outdir}/position_prior_provenance.csv", index=False
    )
    with open(f"{args.outdir}/prior_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "attribution_interval": [args.attrib_lo, args.attrib_hi],
            "min_count": args.min_count,
            "n_attribution_samples": len(sa),
            "floor": 0.01,
            "label": args.eff_col,
            "note": "Single A within attribution interval, not necessarily the entire guide; weights are not measured per-position probabilities."
        }, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.bar(list(prior.keys()), list(prior.values()), color="#4C72B0")
    plt.axvspan(3.5, 8.5, color="red", alpha=0.12, label="edit window 4-8")
    plt.xlabel("protospacer position (1 = PAM-distal)")
    plt.ylabel("normalized A->G activity (peak=1)")
    plt.title("Empirical ABE8e positional prior (single-A attribution)")
    plt.xticks(range(1, 21)); plt.legend(); plt.tight_layout()
    plt.savefig(f"{args.outdir}/fig_empirical_prior.png", dpi=150)
    print(f"\n已保存先验 -> {args.outdir}/empirical_prior.json")


if __name__ == "__main__":
    main()