"""
Narrow search for melt_temperature_target recipe.
Two winners already locked (both Pearson=1.000):
  free_energy           = MFE(grna as RNA)      [ViennaRNA]
  melt_temperature_grna = Tm_NN(grna)           [Biopython]
Target Tm best so far: Tm_NN(sequence 40bp)=0.658 -> try subwindows of `sequence`.
"""

import pandas as pd
import numpy as np
from Bio.SeqUtils import MeltingTemp as mt
from scipy.stats import pearsonr

N = 300


def tm_nn(seq):
    try:
        return float(mt.Tm_NN(seq.upper()))
    except Exception:
        return np.nan


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return np.nan
    return pearsonr(a[m], b[m])[0]


def main():
    df = pd.read_csv("data/processed/train.csv").head(N).reset_index(drop=True)
    db = df["melt_temperature_target"].values.astype(float)

    # locate grna within sequence to define coordinate anchors
    def grna_start(row):
        return str(row["sequence"]).upper().find(str(row["grna"]).upper())
    starts = df.apply(grna_start, axis=1)
    print(f"grna offset within sequence: min={starts.min()} max={starts.max()} "
          f"mode={starts.mode().iloc[0]}\n")

    recipes = {}

    # whole sequence and full_context (baseline references)
    recipes["sequence (40bp full)"] = df["sequence"]
    recipes["full_context (79bp)"] = df["full_context_sequence"]

    # grna itself + grna+PAM style windows
    recipes["grna (20bp)"] = df["grna"]

    def window(row, lo, hi):
        s = str(row["sequence"]).upper()
        return s[lo:hi]

    # sliding fixed-length windows over the 40bp sequence
    for wlen in (20, 23, 25, 30, 33, 35):
        best_name, best_p = None, -1
        for lo in range(0, 40 - wlen + 1):
            vals = df.apply(lambda r: tm_nn(window(r, lo, lo + wlen)), axis=1).values
            p = corr(vals, db)
            if p is not None and p > best_p:
                best_p, best_name = p, f"sequence[{lo}:{lo+wlen}] (len {wlen})"
        recipes[f"__best_len{wlen}__"] = (best_name, best_p)

    # evaluate the direct (non-window) recipes
    print(f"{'recipe':<34}{'Pearson':>10}")
    print("-" * 46)
    for name, val in recipes.items():
        if name.startswith("__best"):
            bn, bp = val
            print(f"{bn:<34}{bp:>10.3f}   <- best fixed window")
        else:
            p = corr(df[name if False else name].apply(tm_nn).values, db) \
                if False else corr(val.apply(tm_nn).values, db)
            print(f"{name:<34}{p:>10.3f}")


if __name__ == "__main__":
    main()