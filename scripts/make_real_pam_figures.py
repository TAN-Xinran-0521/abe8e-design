#!/usr/bin/env python3
"""Real-PAM vs NNN upper-bound off-target comparison figures (HFE vs LMNA).
Reads the two real_vs_nnn_comparison.csv produced by run_real_pam_offtarget.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = __import__('repo_paths').repo_path('results')
HFE  = os.path.join(ROOT, "hfe_offtarget_real/real_vs_nnn_comparison.csv")
LMNA = os.path.join(ROOT, "lmna_offtarget_real/real_vs_nnn_comparison.csv")
OUT  = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

def load(p, site):
    df = pd.read_csv(p)
    df["site"] = site
    df["label"] = df["cas_variant"] + "\n" + df["grna"].str[:8] + "..."
    return df

hfe, lmna = load(HFE, "HFE c.845"), load(LMNA, "LMNA c.1824")

# ---- fig1: real vs NNN grouped bars, log scale, per site ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, df, title in [(axes[0], hfe, "HFE c.845 (clean)"),
                      (axes[1], lmna, "LMNA c.1824 (hard)")]:
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w/2, df["nnn_total_offtargets"], w, label="NNN upper bound",
           color="#B0B0B0")
    ax.bar(x + w/2, df["real_total_offtargets"], w, label="real PAM",
           color="#C44E52")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_title(title)
    for i, r in df.iterrows():
        ax.text(i + w/2, r["real_total_offtargets"]*1.1,
                str(int(r["real_total_offtargets"])), ha="center", fontsize=8)
    ax.legend()
axes[0].set_ylabel("total off-targets (log scale)")
fig.suptitle("Real-PAM vs NNN upper bound off-targets")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_real_vs_nnn.png"), dpi=150)
plt.close(fig)

# ---- fig2: two-site best-candidate head-to-head (real PAM) ----
hfe_best  = hfe.loc[hfe["real_total_offtargets"].idxmin()]
lmna_best = lmna.loc[lmna["real_total_offtargets"].idxmin()]
best = pd.DataFrame([
    {"site": "HFE c.845\n(clean)", "off": hfe_best["real_total_offtargets"],
     "eff": hfe_best["pred_efficiency"], "variant": hfe_best["cas_variant"]},
    {"site": "LMNA c.1824\n(hard)", "off": lmna_best["real_total_offtargets"],
     "eff": lmna_best["pred_efficiency"], "variant": lmna_best["cas_variant"]},
])
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#55A868", "#C44E52"]
bars = ax.bar(best["site"], best["off"], color=colors, width=0.55)
ax.set_yscale("log")
ax.set_ylabel("best-candidate real off-targets (log)")
ax.set_title("Best real-PAM candidate: clean vs hard locus")
for i, r in best.iterrows():
    ax.text(i, r["off"]*1.15,
            f"{int(r['off'])}\n{r['variant']}\neff={r['eff']:.3f}",
            ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_two_site_realpam.png"), dpi=150)
plt.close(fig)

# ---- combined table for the report ----
allrows = pd.concat([hfe, lmna], ignore_index=True)[
    ["site", "rank", "cas_variant", "grna", "pred_efficiency",
     "real_total_offtargets", "nnn_total_offtargets", "real_over_nnn_pct"]]
allrows.to_csv(os.path.join(OUT, "real_pam_offtarget_table.csv"), index=False)

print("=== Combined real-PAM table ===")
print(allrows.to_string(index=False))
print(f"\nHFE best : {hfe_best['cas_variant']} "
      f"{int(hfe_best['real_total_offtargets'])} off-targets, "
      f"eff {hfe_best['pred_efficiency']:.3f}")
print(f"LMNA best: {lmna_best['cas_variant']} "
      f"{int(lmna_best['real_total_offtargets'])} off-targets, "
      f"eff {lmna_best['pred_efficiency']:.3f}")
print(f"Ratio (LMNA/HFE best): "
      f"{lmna_best['real_total_offtargets']/hfe_best['real_total_offtargets']:.1f}x")
print(f"\nFigures -> {OUT}")
