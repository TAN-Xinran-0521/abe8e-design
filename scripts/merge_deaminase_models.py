#!/usr/bin/env python3
"""Merge TadA-8e mutation-scoring results from three models
(ESM2-35M, ESM1v-650M, ESM-IF1) into comparison tables + figures.
pandas 3.0 / matplotlib 3.11 compatible.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = __import__('repo_paths').repo_path('results')
MODELS = {
    "ESM2-35M":   os.path.join(ROOT, "deaminase"),
    "ESM1v-650M": os.path.join(ROOT, "deaminase_650m"),
    "ESM-IF1":    os.path.join(ROOT, "deaminase_if1"),
}
OUT = os.path.join(ROOT, "deaminase_compare")
os.makedirs(OUT, exist_ok=True)

KNOWN = ["A108S", "T110R", "D118N", "H121N", "Y146D", "F148Y", "T165I", "D166N"]


def load_pc():
    """Positive-control delta & rank per model, wide format."""
    frames = {}
    for name, d in MODELS.items():
        p = os.path.join(d, "positive_control.csv")
        if not os.path.exists(p):
            print(f"[warn] missing {p}, skip {name}")
            continue
        df = pd.read_csv(p).set_index("mutation")
        frames[name] = df
    tbl = pd.DataFrame(index=KNOWN)
    for name, df in frames.items():
        tbl[f"{name}_delta"] = df["delta_logp"].reindex(KNOWN)
        tbl[f"{name}_rank"]  = df["rank_in_20aa"].reindex(KNOWN)
    return tbl, frames


def recall_summary(frames):
    rows = []
    for name, df in frames.items():
        valid = df[df["delta_logp"].notna()]
        rows.append({
            "model": name,
            "recall_delta_pos": int((valid["delta_logp"] > 0).sum()),
            "recall_top5": int(valid["top5_recall"].astype(bool).sum()),
            "n": int(len(valid)),
        })
    return pd.DataFrame(rows)


def candidate_consensus():
    """Exact-mutation and site-level agreement across models."""
    all_c = []
    for name, d in MODELS.items():
        p = os.path.join(d, "candidate_mutations.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)[["mutation", "region", "delta_logp"]].copy()
        df["model"] = name
        df["site"] = df["mutation"].str.extract(r"^[A-Z](\d+)").astype(int)
        all_c.append(df)
    if not all_c:
        return pd.DataFrame(), pd.DataFrame()
    cat = pd.concat(all_c, ignore_index=True)

    exact = (cat.groupby("mutation")
                .agg(n_models=("model", "nunique"),
                     models=("model", lambda s: "+".join(sorted(set(s)))),
                     mean_delta=("delta_logp", "mean"),
                     region=("region", "first"))
                .sort_values(["n_models", "mean_delta"], ascending=[False, False])
                .reset_index())
    site = (cat.groupby("site")
               .agg(n_models=("model", "nunique"),
                    models=("model", lambda s: "+".join(sorted(set(s)))),
                    n_candidates=("mutation", "nunique"),
                    region=("region", "first"),
                    max_delta=("delta_logp", "max"))
               .sort_values(["n_models", "max_delta"], ascending=[False, False])
               .reset_index())
    return exact, site


def fig_recall(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w/2, summary["recall_delta_pos"], w, label="delta>0 recall", color="#4C72B0")
    ax.bar(x + w/2, summary["recall_top5"],       w, label="top-5 recall",  color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(summary["model"])
    ax.set_ylabel("recalled known 8e mutations (of 8)")
    ax.set_ylim(0, 8)
    ax.set_title("Positive-control recall across three models")
    for i, r in summary.iterrows():
        ax.text(i - w/2, r["recall_delta_pos"] + 0.1, str(r["recall_delta_pos"]), ha="center")
        ax.text(i + w/2, r["recall_top5"] + 0.1,       str(r["recall_top5"]),       ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_recall_3models.png"), dpi=150)
    plt.close(fig)


def fig_delta(tbl):
    models = [m for m in MODELS if f"{m}_delta" in tbl.columns]
    x = np.arange(len(KNOWN))
    w = 0.8 / len(models)
    colors = {"ESM2-35M": "#4C72B0", "ESM1v-650M": "#55A868", "ESM-IF1": "#C44E52"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, m in enumerate(models):
        vals = tbl[f"{m}_delta"].values.astype(float)
        ax.bar(x + (k - (len(models)-1)/2)*w, vals, w, label=m, color=colors.get(m))
    ax.axhline(0, color="black", lw=0.8)
    # highlight F148Y column
    fi = KNOWN.index("F148Y")
    ax.axvspan(fi - 0.45, fi + 0.45, color="gold", alpha=0.18, zorder=0)
    ax.annotate("F148Y: seq models reject,\nstructure (IF1) supports",
                xy=(fi, 0.5), xytext=(fi - 2.4, 6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xticks(x); ax.set_xticklabels(KNOWN, rotation=0)
    ax.set_ylabel("delta_logp  (mut - wt)")
    ax.set_title("Known 8e mutations: per-model preference (delta_logp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_delta_known_3models.png"), dpi=150)
    plt.close(fig)


def main():
    tbl, frames = load_pc()
    tbl.to_csv(os.path.join(OUT, "positive_control_3models.csv"))
    print("=== Positive control (3 models) ===")
    print(tbl.round(3).to_string())

    summary = recall_summary(frames)
    summary.to_csv(os.path.join(OUT, "recall_summary.csv"), index=False)
    print("\n=== Recall summary ===")
    print(summary.to_string(index=False))

    exact, site = candidate_consensus()
    exact.to_csv(os.path.join(OUT, "candidate_consensus_exact.csv"), index=False)
    site.to_csv(os.path.join(OUT, "candidate_consensus_site.csv"), index=False)
    print("\n=== Candidate consensus: exact mutations in >=2 models ===")
    multi = exact[exact["n_models"] >= 2]
    print(multi.to_string(index=False) if not multi.empty else "(none shared exactly)")
    print("\n=== Candidate consensus: sites hit in >=2 models ===")
    multis = site[site["n_models"] >= 2]
    print(multis.to_string(index=False) if not multis.empty else "(none)")

    fig_recall(summary)
    fig_delta(tbl)
    print(f"\nFigures + tables written to {OUT}")


if __name__ == "__main__":
    main()
