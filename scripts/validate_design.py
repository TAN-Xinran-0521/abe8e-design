"""
Validate the sgRNA design tool on REAL sequences from the held-out test set.

Two things it shows:
  1. Efficiency validation: predicted efficiency vs reported efficiency
     on real ABE8e targets (correlation on real data).
  2. Design-tool sanity check: given a real context + target A, does the
     PAM-scan pipeline recover the real grna among its candidates?

Robust to strand: locates the known grna in full_context_sequence by
matching both forward and reverse-complement.
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr, pearsonr

# reuse the design tool's functions
from design_sgrna import (
    sequence_to_onehot, find_candidates, build_feature_lookup,
    ENERGY_COLS, PHYS_COLS,
)

TEST_CSV = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_baseline.pkl"
CONTEXT_COL = "full_context_sequence"
GRNA_COL = "grna"
LABEL_COL = "efficiency_full_grna_reported"


def revcomp(s):
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(s))


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def featurize_row(row):
    """107-dim feature vector for a real test-set row (has all energy/phys)."""
    seq = sequence_to_onehot(row[GRNA_COL])          # 80
    energy = row[ENERGY_COLS].values.astype(float)   # 24
    phys = row[PHYS_COLS].values.astype(float)       # 3
    return np.concatenate([seq, energy, phys])


def main():
    df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df)} test rows.")
    print(f"Columns with 'seq'/'context': "
          f"{[c for c in df.columns if 'seq' in c.lower() or 'context' in c.lower()]}\n")

    model = load_model(MODEL_PATH)

    # ---- 1. Efficiency validation on real sequences ----
    X = np.vstack([featurize_row(r) for _, r in df.iterrows()])
    y_true = df[LABEL_COL].values
    y_pred = np.clip(model.predict(X), 0.0, 1.0)

    sp = spearmanr(y_true, y_pred)[0]
    pe = pearsonr(y_true, y_pred)[0]
    mae = np.mean(np.abs(y_true - y_pred))
    print("=== Efficiency validation (real test sequences) ===")
    print(f"Spearman: {sp:.3f} | Pearson: {pe:.3f} | MAE: {mae:.3f}")
    print("(should match your baseline Test Spearman ~0.751)\n")

    # show a few concrete predicted-vs-real examples
    df_show = df.copy()
    df_show["pred"] = y_pred
    df_show = df_show.sort_values(LABEL_COL, ascending=False)
    print("Top real-efficiency targets (real vs predicted):")
    print(f"{'grna':<24}{'real':>7}{'pred':>7}")
    for _, r in df_show.head(5).iterrows():
        print(f"{r[GRNA_COL]:<24}{r[LABEL_COL]:>7.3f}{r['pred']:>7.3f}")

    # ---- 2. Design-tool sanity check on one real context ----
    print("\n=== Design-tool recovery check (one real context) ===")
    row = df.iloc[0]
    context = str(row[CONTEXT_COL]).upper()
    grna = str(row[GRNA_COL]).upper()

    # locate grna in context (forward or reverse strand)
    strand, idx = None, context.find(grna)
    if idx >= 0:
        strand = "+"
    else:
        rc = revcomp(grna)
        idx = context.find(rc)
        if idx >= 0:
            strand = "-"

    if strand is None:
        print(f"grna not found as substring of context (may be padded/trimmed).")
        print(f"  grna:    {grna}")
        print(f"  context: {context}")
        return

    print(f"Real grna located on {strand} strand at context index {idx}.")
    # first A inside the protospacer's edit window (positions 4-8, 1-based)
    proto = grna if strand == "+" else revcomp(grna)
    window_as = [p for p in range(4, 9) if proto[p - 1] == "A"]
    print(f"A positions in edit window (4-8) of the real grna: {window_as}")

    if strand == "+" and window_as:
        target_pos = idx + (window_as[0] - 1)   # 0-based in context
        print(f"Running design tool with target_pos={target_pos} "
              f"(context[{target_pos}]={context[target_pos]})")
        cands = find_candidates(context, target_pos)
        found = any(c["grna"] == grna for c in cands)
        print(f"Candidates found: {len(cands)} | real grna recovered: {found}")
    else:
        print("(reverse-strand target: PAM scan runs on the given strand only; "
              "design tool would need the reverse strand for this locus.)")


if __name__ == "__main__":
    main()