# HISTORICAL DATABASE-RECIPE MODULE; not used for B-prime inference.
"""
Physical feature computation for ABE8e gRNA design.

Recipes reverse-engineered from BE-dataHIVE (all Pearson=1.000 vs DB columns,
see calibrate_features.py / calibrate_target_tm.py):

  free_energy             = ViennaRNA MFE of gRNA treated as RNA (T->U)
  melt_temperature_grna   = Biopython Tm_NN(gRNA)
  melt_temperature_target = Biopython Tm_NN(sequence[10:30])   # 20bp target window

These let NEW targets get REAL physical features instead of mean-imputation,
matching the training-data feature distribution exactly.

NOTE: energy_1..24 are NOT reproducible from sequence (definition not public);
those remain mean-imputed for novel targets.
"""

import RNA
from Bio.SeqUtils import MeltingTemp as mt

TARGET_WINDOW = (10, 30)  # sequence[10:30], calibrated Pearson=1.000


def free_energy_grna(grna):
    """ViennaRNA MFE (kcal/mol) of the gRNA spacer treated as RNA."""
    rna = grna.upper().replace("T", "U")
    fc = RNA.fold_compound(rna)
    _, mfe = fc.mfe()
    return float(mfe)


def melt_temp_grna(grna):
    """Nearest-neighbor Tm (deg C) of the gRNA."""
    return float(mt.Tm_NN(grna.upper()))


def melt_temp_target(sequence):
    """
    Nearest-neighbor Tm (deg C) of the 20bp target window sequence[10:30].
    `sequence` is the 40bp local context column from BE-dataHIVE.
    """
    window = sequence.upper()[TARGET_WINDOW[0]:TARGET_WINDOW[1]]
    return float(mt.Tm_NN(window))


def compute_physical(grna, sequence):
    """
    Return the 3 physical features in the SAME ORDER as the training columns:
      [free_energy, melt_temperature_grna, melt_temperature_target]
    """
    return [
        free_energy_grna(grna),
        melt_temp_grna(grna),
        melt_temp_target(sequence),
    ]


if __name__ == "__main__":
    # verify recipes reproduce DB columns on a few real rows
    import pandas as pd
    df = pd.read_csv("data/processed/train.csv").head(5)
    print(f"{'grna':<22}{'fe_calc':>9}{'fe_db':>9}"
          f"{'tmg_calc':>10}{'tmg_db':>9}{'tmt_calc':>10}{'tmt_db':>9}")
    print("-" * 78)
    for _, r in df.iterrows():
        fe, tmg, tmt = compute_physical(r["grna"], r["sequence"])
        print(f"{r['grna']:<22}{fe:>9.2f}{r['free_energy']:>9.2f}"
              f"{tmg:>10.2f}{r['melt_temperature_grna']:>9.2f}"
              f"{tmt:>10.2f}{r['melt_temperature_target']:>9.2f}")