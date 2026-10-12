"""
Predicted vs real efficiency, on real test-set targets (plus & minus strand).

For each strand it:
  1. picks a real context whose target A sits in the edit window,
  2. runs the design tool's candidate scan + efficiency prediction,
  3. matches the REAL grna among the candidates,
  4. prints real reported efficiency vs model-predicted efficiency.
"""

import pandas as pd

from design_sgrna import (
    find_candidates, score_candidates,
    load_efficiency_model, build_feature_lookup,
)

TEST_CSV = "data/processed/test.csv"
MODEL_PATH = "models/xgboost_baseline.pkl"
PROCESSED_CSV = "data/processed/train.csv"
CONTEXT_COL = "full_context_sequence"
GRNA_COL = "grna"
LABEL_COL = "efficiency_full_grna_reported"
WINDOW = (4, 8)
WEIGHTS = {"w1": 0.4, "w2": 0.3, "w3": 0.3}  # therapy

_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def revcomp(s):
    return "".join(_COMP.get(b, "N") for b in reversed(s))


def find_example(df, want_strand):
    """Return (context, target_pos, real_grna, real_eff) for the wanted strand."""
    for _, r in df.iterrows():
        context = str(r[CONTEXT_COL]).upper()
        grna = str(r[GRNA_COL]).upper()
        if len(grna) != 20:
            continue
        if want_strand == "+":
            idx = context.find(grna)
            if idx < 0:
                continue
            for p in range(WINDOW[0], WINDOW[1] + 1):
                if grna[p - 1] == "A":
                    target_pos = idx + (p - 1)
                    if context[target_pos] == "A":
                        return context, target_pos, grna, float(r[LABEL_COL])
        else:
            rc = revcomp(grna)
            idx = context.find(rc)
            if idx < 0:
                continue
            for p in range(WINDOW[0], WINDOW[1] + 1):
                if grna[p - 1] == "A":
                    target_pos = idx + (20 - p)
                    if 0 <= target_pos < len(context) and context[target_pos] == "T":
                        return context, target_pos, grna, float(r[LABEL_COL])
    return None


def main():
    df = pd.read_csv(TEST_CSV)
    model = load_efficiency_model(MODEL_PATH)
    feat_lookup = build_feature_lookup(PROCESSED_CSV)

    print("=" * 74)
    print(f"{'strand':<8}{'real_grna':<24}{'real_eff':>10}{'pred_eff':>10}{'|diff|':>9}")
    print("=" * 74)

    for strand in ("+", "-"):
        res = find_example(df, strand)
        if res is None:
            print(f"{strand:<8}(no example found)")
            continue
        context, target_pos, real_grna, real_eff = res

        cands = find_candidates(context, target_pos)
        scored = score_candidates(cands, model, WEIGHTS, feat_lookup)

        row = scored[scored["grna"] == real_grna] if not scored.empty else None
        if row is None or row.empty:
            print(f"{strand:<8}{real_grna:<24}{real_eff:>10.3f}"
                  f"{'not recovered':>19}")
            continue

        pred = float(row["pred_efficiency"].iloc[0])
        print(f"{strand:<8}{real_grna:<24}{real_eff:>10.3f}{pred:>10.3f}"
              f"{abs(real_eff - pred):>9.3f}")

    print("=" * 74)
    print("real_eff : reported efficiency from BE-dataHIVE (ground truth)")
    print("pred_eff : XGBoost prediction for the SAME grna via the design tool")
    print("A single-point |diff| reflects the model's absolute error at this locus;")
    print("ranking quality across the whole test set is Spearman 0.751 (see baseline).")


if __name__ == "__main__":
    main()