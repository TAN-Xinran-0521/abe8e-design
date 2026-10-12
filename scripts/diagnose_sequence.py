"""
Diagnose the 40bp `sequence` column structure vs grna, so we know how to
compute melt_temperature_target for NOVEL design candidates (which have no
standard 40bp sequence column).
"""

import pandas as pd
import numpy as np
from Bio.SeqUtils import MeltingTemp as mt
from scipy.stats import pearsonr

df = pd.read_csv("data/processed/train.csv").head(2000)

print("=== 1. sample rows: sequence / grna / sequence[10:30] ===")
for _, r in df.head(5).iterrows():
    seq = str(r["sequence"]).upper()
    grna = str(r["grna"]).upper()
    print(f"grna         : {grna}")
    print(f"sequence     : {seq}")
    print(f"sequence[10:30]: {seq[10:30]}")
    print(f"grna in seq? offset = {seq.find(grna)}")
    print("-" * 60)

print("\n=== 2. where does grna sit inside sequence? ===")
offsets = df.apply(lambda r: str(r["sequence"]).upper().find(str(r["grna"]).upper()), axis=1)
print(offsets.value_counts().head(10))

print("\n=== 3. approximation quality for target Tm ===")
def tm(s):
    try: return float(mt.Tm_NN(s.upper()))
    except: return np.nan

db_tt = df["melt_temperature_target"].values.astype(float)
tm_grna = df["grna"].apply(tm).values
tm_win  = df["sequence"].apply(lambda s: tm(str(s)[10:30])).values

def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return pearsonr(a[m], b[m])[0]

print(f"Tm_NN(grna)          vs DB target Tm : Pearson {corr(tm_grna, db_tt):.3f}")
print(f"Tm_NN(sequence[10:30]) vs DB target Tm : Pearson {corr(tm_win, db_tt):.3f}")
print(f"Tm_NN(grna) vs Tm_NN(sequence[10:30])  : Pearson {corr(tm_grna, tm_win):.3f}")