"""
Calibrate re-computed physical features against BE-dataHIVE's existing columns.

Goal: find a self-consistent recipe (tool + input sequence) whose output
correlates highly with the database's free_energy / melt_temperature_* columns.
If correlation is high, NEW targets can be computed on the fly and fed to the
existing full-feature model (0.751) WITHOUT retraining and WITHOUT dropping energy_1..24.

Tries multiple input choices to reverse-engineer which sequence the DB used.
"""

import pandas as pd
import numpy as np
import RNA
from Bio.SeqUtils import MeltingTemp as mt
from scipy.stats import pearsonr, spearmanr

N = 300  # sample rows for speed


def mfe(seq, as_rna=True):
    s = seq.upper().replace("T", "U") if as_rna else seq.upper()
    fc = RNA.fold_compound(s)
    _, e = fc.mfe()
    return float(e)


def tm_nn(seq):
    try:
        return float(mt.Tm_NN(seq.upper()))
    except Exception:
        return np.nan


def tm_gc(seq):
    try:
        return float(mt.Tm_GC(seq.upper()))
    except Exception:
        return np.nan


def tm_wallace(seq):
    try:
        return float(mt.Tm_Wallace(seq.upper()))
    except Exception:
        return np.nan


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return np.nan, np.nan
    return pearsonr(a[m], b[m])[0], spearmanr(a[m], b[m])[0]


def main():
    df = pd.read_csv("data/processed/train.csv").head(N).reset_index(drop=True)
    print(f"Calibrating on {len(df)} rows.\n")

    # ---- free_energy : which sequence + RNA/DNA? ----
    print("=" * 68)
    print("free_energy  (DB column) vs recomputed MFE")
    print("=" * 68)
    db_fe = df["free_energy"].values.astype(float)
    candidates_fe = {
        "MFE(grna, RNA)":        df["grna"].apply(lambda s: mfe(s, True)).values,
        "MFE(grna, DNA)":        df["grna"].apply(lambda s: mfe(s, False)).values,
        "MFE(sequence, RNA)":    df["sequence"].apply(lambda s: mfe(s, True)).values,
        "MFE(full_context,RNA)": df["full_context_sequence"].apply(lambda s: mfe(s, True)).values,
    }
    print(f"{'recipe':<26}{'Pearson':>10}{'Spearman':>10}")
    print("-" * 68)
    for name, vals in candidates_fe.items():
        p, s = corr(vals, db_fe)
        print(f"{name:<26}{p:>10.3f}{s:>10.3f}")

    # ---- melt_temperature_grna ----
    print("\n" + "=" * 68)
    print("melt_temperature_grna  (DB column) vs recomputed Tm")
    print("=" * 68)
    db_tg = df["melt_temperature_grna"].values.astype(float)
    cand_tg = {
        "Tm_NN(grna)":      df["grna"].apply(tm_nn).values,
        "Tm_GC(grna)":      df["grna"].apply(tm_gc).values,
        "Tm_Wallace(grna)": df["grna"].apply(tm_wallace).values,
    }
    print(f"{'recipe':<26}{'Pearson':>10}{'Spearman':>10}")
    print("-" * 68)
    for name, vals in cand_tg.items():
        p, s = corr(vals, db_tg)
        print(f"{name:<26}{p:>10.3f}{s:>10.3f}")

    # ---- melt_temperature_target : which sequence? ----
    print("\n" + "=" * 68)
    print("melt_temperature_target  (DB column) vs recomputed Tm")
    print("=" * 68)
    db_tt = df["melt_temperature_target"].values.astype(float)
    cand_tt = {
        "Tm_NN(sequence 40bp)":     df["sequence"].apply(tm_nn).values,
        "Tm_NN(full_context 79bp)": df["full_context_sequence"].apply(tm_nn).values,
        "Tm_GC(sequence 40bp)":     df["sequence"].apply(tm_gc).values,
        "Tm_GC(full_context 79bp)": df["full_context_sequence"].apply(tm_gc).values,
    }
    print(f"{'recipe':<28}{'Pearson':>10}{'Spearman':>10}")
    print("-" * 68)
    for name, vals in cand_tt.items():
        p, s = corr(vals, db_tt)
        print(f"{name:<28}{p:>10.3f}{s:>10.3f}")

    print("\nInterpretation:")
    print(" Pearson > 0.95 -> recipe matches DB definition; use it for new targets,")
    print("                   feed into existing full-feature model, NO retraining.")
    print(" 0.7-0.95       -> correlated but not identical; retrain on recomputed recipe.")
    print(" < 0.7          -> definitions differ; drop these, retrain seq+recomputed.")


if __name__ == "__main__":
    main()