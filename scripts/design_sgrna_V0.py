"""
Candidate sgRNA design tool for ABE8e (dual-strand, multi-objective).

Efficiency model: B' design model (models/xgboost_design_v2.pkl), a
self-consistent recipe with NO energy features and NO mean-imputation.
All 3 physical features are computed from the 20nt protospacer alone,
IDENTICAL to how B' was trained -> zero train/inference gap on novel targets.

  free_energy             = MFE(protospacer as RNA)  [ViennaRNA]
  melt_temperature_grna   = Tm_NN(protospacer)       [Biopython]
  melt_temperature_target = Tm_NN(protospacer)       [same 20nt, self-consistent]

Feature vector per candidate: 80 one-hot + 3 physical = 83 dims.

Semantics (pathogenic-mutation correction, dual-strand):
  context[target_pos] == 'A' -> target A on plus strand  -> design on '+'
  context[target_pos] == 'T' -> target A on minus strand -> design on '-'
  context[target_pos] in CG  -> error (ABE edits A only)

Pipeline:
  pick strand -> scan NGG PAM -> 20nt protospacer
  -> keep candidates placing target A in edit window (pos 4-8)
  -> B' efficiency prediction (physical features recomputed from protospacer)
  -> bystander risk (extra A in window, rule-based)
  -> off-target heuristic (offtarget_heuristic.py)
  -> 3-objective composite score -> Pareto flag -> Top-K

Composite: Score = w1*efficiency - w2*bystander_norm - w3*offtarget_risk
Presets: research 0.8/0.1/0.1, therapy 0.4/0.3/0.3, clinical 0.3/0.4/0.3
"""

import re
import pickle
import argparse
import numpy as np
import pandas as pd
import RNA
from Bio.SeqUtils import MeltingTemp as mt

from offtarget_heuristic import offtarget_risk

# ------------------------------------------------------------------ config
PROTOSPACER_LEN = 20
EDIT_WINDOW = (4, 8)          # 1-based inclusive protospacer positions
BYSTANDER_NORM = 4.0

WEIGHT_PRESETS = {
    "research": {"w1": 0.8, "w2": 0.1, "w3": 0.1},
    "therapy":  {"w1": 0.4, "w2": 0.3, "w3": 0.3},
    "clinical": {"w1": 0.3, "w2": 0.4, "w3": 0.3},
}

_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


# ------------------------------------------------------------------ seq utils
def reverse_complement(seq):
    return "".join(_COMP.get(b, "N") for b in reversed(seq))


def sequence_to_onehot(seq):
    """Must match training encoding (build_features.py / B')."""
    mapping = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0],
               "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}
    out = []
    for n in seq.upper():
        out.extend(mapping.get(n, [0, 0, 0, 0]))
    return out


def phys_from_grna(grna):
    """
    B' recipe: all 3 physical features from the 20nt protospacer alone.
    IDENTICAL to train_design_model_v2.py -> guarantees口径 consistency.
      [free_energy(MFE as RNA), Tm_NN(grna), Tm_NN(grna)]
    """
    g = grna.upper()
    rna = g.replace("T", "U")
    fc = RNA.fold_compound(rna)
    _, mfe = fc.mfe()
    tm_g = float(mt.Tm_NN(g))
    return [float(mfe), tm_g, tm_g]


# ------------------------------------------------------------------ PAM scan
def _scan_strand_for_target(seq, target_idx, strand):
    """Scan `seq` for NGG PAM; keep candidates with target A in edit window."""
    out = []
    pam_re = re.compile(r"(?=([ACGT]GG))")  # overlapping NGG
    for m in pam_re.finditer(seq):
        pam_start = m.start(1)
        pam = m.group(1)
        proto_start = pam_start - PROTOSPACER_LEN
        proto_end = pam_start
        if proto_start < 0:
            continue
        protospacer = seq[proto_start:proto_end]
        if len(protospacer) != PROTOSPACER_LEN:
            continue
        target_in_proto = target_idx - proto_start + 1  # 1-based
        if not (EDIT_WINDOW[0] <= target_in_proto <= EDIT_WINDOW[1]):
            continue
        if protospacer[target_in_proto - 1] != "A":
            continue
        out.append({
            "grna": protospacer,
            "pam": pam,
            "target_pos_in_proto": target_in_proto,
            "strand": strand,
        })
    return out


def find_candidates(context, target_pos):
    """Auto-select the strand carrying the target A, then scan NGG PAM there."""
    context = context.upper()
    base = context[target_pos]
    if base == "A":
        return _scan_strand_for_target(context, target_pos, "+")
    if base == "T":
        rc = reverse_complement(context)
        L = len(context)
        return _scan_strand_for_target(rc, (L - 1) - target_pos, "-")
    raise ValueError(
        f"Base at target_pos {target_pos} is '{base}'. ABE edits A only; "
        f"target base must be 'A' (plus) or 'T' (minus)."
    )


def bystander_a_count(protospacer, target_pos_in_proto, window=EDIT_WINDOW):
    lo, hi = window
    count = 0
    for i in range(lo, hi + 1):
        if i == target_pos_in_proto:
            continue
        if protospacer[i - 1] == "A":
            count += 1
    return count


# ------------------------------------------------------------------ efficiency
def load_efficiency_model(model_path):
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def predict_efficiency(model, candidates):
    """
    B' feature policy: 80 one-hot + 3 physical recomputed from the protospacer.
    No energy, no lookup, no mean-imputation -> identical to training recipe.
    """
    rows = []
    for c in candidates:
        rows.append(sequence_to_onehot(c["grna"]) + phys_from_grna(c["grna"]))
    X = np.array(rows, dtype=np.float32)
    return np.clip(model.predict(X), 0.0, 1.0)


# ------------------------------------------------------------------ scoring
def score_candidates(candidates, model, weights):
    if not candidates:
        return pd.DataFrame()
    effs = predict_efficiency(model, candidates)
    records = []
    for c, eff in zip(candidates, effs):
        by_count = bystander_a_count(c["grna"], c["target_pos_in_proto"])
        by_norm = min(1.0, by_count / BYSTANDER_NORM)
        ot_score, ot_parts = offtarget_risk(c["grna"])
        composite = (weights["w1"] * float(eff)
                     - weights["w2"] * by_norm
                     - weights["w3"] * ot_score)
        records.append({
            "grna": c["grna"],
            "pam": c["pam"],
            "strand": c["strand"],
            "target_pos": c["target_pos_in_proto"],
            "pred_efficiency": round(float(eff), 3),
            "bystander_A": by_count,
            "offtarget_risk": ot_score,
            "ot_seed": ot_parts["seed"],
            "ot_gc": ot_parts["gc"],
            "ot_homopolymer": ot_parts["homopolymer"],
            "ot_complexity": ot_parts["complexity"],
            "composite_score": round(composite, 3),
        })
    df = pd.DataFrame(records)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def pareto_front(df):
    if df.empty:
        return df
    eff = df["pred_efficiency"].values
    by = df["bystander_A"].values
    ot = df["offtarget_risk"].values
    n = len(df)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ge = (eff[j] >= eff[i] and by[j] <= by[i] and ot[j] <= ot[i])
            sb = (eff[j] > eff[i] or by[j] < by[i] or ot[j] < ot[i])
            if ge and sb:
                dominated[i] = True
                break
    out = df.copy()
    out["pareto_optimal"] = ~dominated
    return out


# ------------------------------------------------------------------ reporting
def print_report(df, scenario, weights, top_k, target_base, strand_used):
    print("\n" + "=" * 82)
    print(f"=== ABE8e candidate sgRNA ranking | scenario: {scenario} "
          f"(w1={weights['w1']} w2={weights['w2']} w3={weights['w3']}) ===")
    print(f"=== target base on plus strand: '{target_base}'  ->  "
          f"designing on {strand_used} strand | model: B' (seq+phys, no energy) ===")
    print("=" * 82)
    if df.empty:
        print("No candidates found (no NGG PAM places the target A in window 4-8).")
        return
    show = ["grna", "pam", "strand", "target_pos", "pred_efficiency",
            "bystander_A", "offtarget_risk", "composite_score"]
    if "pareto_optimal" in df.columns:
        show.append("pareto_optimal")
    print(df.head(top_k)[show].to_string())
    print("\n--- notes ---")
    print("model           : B' design model (80 one-hot + 3 physical, no energy)")
    print("                  physical features recomputed from protospacer (self-consistent)")
    print("strand          : + = target A on plus strand, - = on minus strand")
    print("pred_efficiency : B' output (0-1, edited fraction); novel-target Test Spearman ~0.67")
    print("target_pos      : target A position in protospacer (optimal window 4-8)")
    print("bystander_A     : extra A's in window excl. target (rule-based)")
    print("offtarget_risk  : heuristic intrinsic specificity risk (0-1, higher=worse)")
    print("composite_score : w1*eff - w2*bystander_norm - w3*offtarget_risk")
    print("pareto_optimal  : non-dominated across (eff up, bystander down, ot down)")
    print("\n[!] offtarget_risk is a sequence heuristic, NOT genome alignment.")
    print("    Real off-target spectrum needs Cas-OFFinder/CRISPOR + wet-lab (GUIDE-seq).")
    print("    Cas9-independent (deaminase) off-target is not captured here.")


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(
        description="ABE8e candidate sgRNA design (dual-strand, multi-objective, B' model)."
    )
    parser.add_argument("--context", required=True,
                        help="Plus-strand sequence containing the target base.")
    parser.add_argument("--target_pos", type=int, required=True,
                        help="0-based index of the target base within --context.")
    parser.add_argument("--model", default="models/xgboost_design_v2.pkl",
                        help="B' efficiency model (default: models/xgboost_design_v2.pkl).")
    parser.add_argument("--scenario", choices=list(WEIGHT_PRESETS.keys()),
                        default="therapy")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    context = args.context.upper().strip()
    if args.target_pos < 0 or args.target_pos >= len(context):
        raise ValueError("target_pos out of range for the given context.")
    target_base = context[args.target_pos]
    if target_base not in ("A", "T"):
        raise ValueError(
            f"Base at target_pos {args.target_pos} is '{target_base}'. "
            f"target base must be 'A' (plus) or 'T' (minus).")
    strand_used = "+" if target_base == "A" else "-"
    weights = WEIGHT_PRESETS[args.scenario]

    print(f"Loading B' efficiency model: {args.model}")
    model = load_efficiency_model(args.model)

    print(f"Target base '{target_base}' at index {args.target_pos} "
          f"-> scanning {strand_used} strand (context length {len(context)})...")
    candidates = find_candidates(context, args.target_pos)
    print(f"Found {len(candidates)} candidate(s) with target A in window 4-8.")

    df = score_candidates(candidates, model, weights)
    if not df.empty:
        df = pareto_front(df)
    print_report(df, args.scenario, weights, args.top_k, target_base, strand_used)

    if args.out_csv and not df.empty:
        df.to_csv(args.out_csv)
        print(f"\nSaved full ranked table -> {args.out_csv}")


if __name__ == "__main__":
    main()