"""
Find gRNAs shared between Study 3 and Study 4.
Directly compare reported (Study3) vs calculated (Study4) efficiency
for the SAME gRNA. This isolates the label-definition problem.
"""

import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Load full raw data (has both studies)
df = pd.read_csv("data/raw/efficiency_ABE8e_raw.csv")

s3 = df[df["study_id"] == 3][["grna", "efficiency_full_grna_reported"]].dropna()
s4 = df[df["study_id"] == 4][["grna", "efficiency_full_grna_calculated"]].dropna()

print(f"Study 3 gRNAs: {s3['grna'].nunique()} unique")
print(f"Study 4 gRNAs: {s4['grna'].nunique()} unique")

# Find shared gRNAs
shared = set(s3["grna"]) & set(s4["grna"])
print(f"\nShared gRNAs between studies: {len(shared)}")

if len(shared) == 0:
    print("\nNO SHARED gRNAs. The two studies target completely different sites.")
    print("This means we cannot directly compare labels on identical sequences.")
    print("The negative correlation likely reflects BOTH:")
    print("  1. Different gRNA populations (different target sites)")
    print("  2. Different label definitions (reported vs calculated)")
else:
    # Merge on shared gRNAs, averaging duplicates
    s3_avg = s3.groupby("grna")["efficiency_full_grna_reported"].mean()
    s4_avg = s4.groupby("grna")["efficiency_full_grna_calculated"].mean()

    merged = pd.DataFrame({
        "reported_study3": s3_avg,
        "calculated_study4": s4_avg
    }).dropna()

    print(f"\nMatched pairs: {len(merged)}")
    print("\n=== Direct label comparison on SAME gRNAs ===")
    sp = spearmanr(merged["reported_study3"], merged["calculated_study4"])[0]
    pr = pearsonr(merged["reported_study3"], merged["calculated_study4"])[0]
    print(f"Spearman: {sp:.4f}")
    print(f"Pearson:  {pr:.4f}")
    print("\nSample pairs:")
    print(merged.head(15).round(4).to_string())