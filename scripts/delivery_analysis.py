#!/usr/bin/env python3
"""Size & AAV-delivery adaptation analysis for ABE8e + Cas9 editors.
Outputs structured tables for the report appendix.
NOTE: sizes are widely-cited approximate coding lengths; verify against a
primary reference before quoting exact numbers in the report.
"""
import os, json
import pandas as pd

OUT = __import__('repo_paths').repo_path('results/delivery')
os.makedirs(OUT, exist_ok=True)

AAV_CARGO_BP = 4700  # practical single-AAV packaging limit (~4.7 kb incl. ITRs/promoter/polyA)

# ---- component coding sizes (approx, in bp of CDS) ----
components = [
    # name, coding_bp, note
    ("SpCas9 (nCas9, D10A nickase)", 4101, "~1368 aa; dominates the payload"),
    ("TadA-8e deaminase (monomer)",   498, "166 aa; single-monomer ABE8e"),
    ("Linkers + BP-NLS x2",           300, "32aa linker + NLS elements"),
    ("sgRNA cassette (U6 + scaffold + 20nt)", 500, "promoter + scaffold + spacer"),
    ("Minimal promoter (e.g. EFS)",   400, "short pol II promoter"),
    ("polyA signal",                  150, "e.g. SV40 late polyA (short)"),
]
comp_df = pd.DataFrame(components, columns=["component", "coding_bp", "note"])
editor_core = comp_df.loc[comp_df["component"].str.contains("Cas9|TadA|Linker"),
                          "coding_bp"].sum()
full_cassette = comp_df["coding_bp"].sum()

# ---- alternative compact chassis ----
alternatives = [
    ("SpCas9-ABE8e (standard)",  full_cassette, "exceeds single AAV", "split-AAV / dual-vector"),
    ("SaCas9-ABE8e",             full_cassette - 900, "SaCas9 ~3159bp (~1053aa), smaller",
     "borderline; needs minimal regulatory parts"),
    ("Un1Cas12f1-ABE (miniBE)",  498 + 1400 + 800, "Cas12f1 ~1500bp (~500aa) ultra-small",
     "fits single AAV; STUminiBE-type design"),
    ("CasMINI / enAsCas12f",     498 + 1450 + 800, "engineered miniature effector",
     "single-AAV feasible"),
]
alt_df = pd.DataFrame(alternatives,
                      columns=["chassis", "approx_total_bp", "size_note", "delivery_strategy"])
alt_df["fits_single_aav"] = alt_df["approx_total_bp"] <= AAV_CARGO_BP

# ---- delivery strategy summary ----
strategies = [
    ("Split-AAV (intein)", "Split editor across 2 AAVs; intein-mediated protein trans-splicing reconstitutes full editor in vivo.",
     "Proven for SpCas9-BE; ~2x viral dose, reconstitution efficiency varies."),
    ("Dual-AAV (split-intein N/C)", "N- and C-terminal halves on separate vectors.",
     "Standard workaround for >4.7kb editors."),
    ("Miniature Cas (Cas12f/CasMINI)", "Replace SpCas9 with ~500aa effector.",
     "Single-AAV feasible; may trade off activity/PAM range."),
    ("Deaminase truncation/engineering", "Trim non-essential TadA regions; our ESM candidates target stability/specificity.",
     "Marginal size gain; main value is specificity, not size."),
    ("Non-viral (LNP mRNA/RNP)", "Deliver editor as mRNA or RNP in lipid nanoparticles.",
     "No cargo-size limit; transient expression lowers off-target exposure."),
]
strat_df = pd.DataFrame(strategies, columns=["strategy", "mechanism", "tradeoff"])

comp_df.to_csv(os.path.join(OUT, "component_sizes.csv"), index=False)
alt_df.to_csv(os.path.join(OUT, "compact_chassis_options.csv"), index=False)
strat_df.to_csv(os.path.join(OUT, "delivery_strategies.csv"), index=False)

summary = {
    "aav_single_cargo_limit_bp": AAV_CARGO_BP,
    "editor_core_bp (Cas9+TadA+linker)": int(editor_core),
    "full_cassette_bp (incl. sgRNA+promoter+polyA)": int(full_cassette),
    "exceeds_single_aav": bool(full_cassette > AAV_CARGO_BP),
    "overflow_bp": int(full_cassette - AAV_CARGO_BP),
    "conclusion": ("Standard SpCas9-ABE8e cassette (~{} bp) exceeds the ~4.7kb single-AAV "
                   "limit, so requires split/dual-AAV or a miniature-Cas chassis. Deaminase "
                   "engineering (our ESM candidates) improves specificity, not delivery size."
                   ).format(int(full_cassette)),
}
with open(os.path.join(OUT, "delivery_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("=== Component sizes ===")
print(comp_df.to_string(index=False))
print(f"\nEditor core (Cas9+TadA+linker): {int(editor_core)} bp")
print(f"Full cassette: {int(full_cassette)} bp  | single-AAV limit: {AAV_CARGO_BP} bp")
print(f"Overflow: {int(full_cassette - AAV_CARGO_BP)} bp  -> single AAV NOT sufficient")
print("\n=== Compact chassis options ===")
print(alt_df.to_string(index=False))
print("\n=== Delivery strategies ===")
print(strat_df.to_string(index=False))
print(f"\nOutputs -> {OUT}")
