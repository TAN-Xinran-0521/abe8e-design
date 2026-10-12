"""
Candidate sgRNA design tool for ABE8e (dual-strand, multi-objective, PAM-variant aware).

Efficiency model: B' design model (models/xgboost_design_v2.pkl), a
self-consistent recipe with NO energy features and NO mean-imputation.
All 3 physical features are computed from the 20nt protospacer alone,
IDENTICAL to how B' was trained -> zero train/inference gap on novel targets.

  free_energy             = MFE(protospacer as RNA)  [ViennaRNA]
  melt_temperature_grna   = Tm_NN(protospacer)       [Biopython]
  melt_temperature_target = Tm_NN(protospacer)       [same 20nt, self-consistent]

Feature vector per candidate: 80 one-hot + 3 physical = 83 dims.

Conformal prediction interval (optional, --conformal):
  Reads conformal_predict.py output (split-conformal absolute-residual half-width
  q_hat calibrated on val). Adds pred_eff_low / pred_eff_high columns
  = clip(pred_eff -/+ q, 0, 1). Point estimate still drives ranking; the interval
  is附加可信度信息. 80% coverage ~0.80, 90% coverage ~0.896 (near-nominal).

Semantics (pathogenic-mutation correction, dual-strand):
  context[target_pos] == 'A' -> target A on plus strand  -> design on '+'
  context[target_pos] == 'T' -> target A on minus strand -> design on '-'
  context[target_pos] in CG  -> error (ABE edits A only)

PAM variants (expanding recognition range; addresses "editing window / PAM" bottleneck):
  SpCas9     NGG   wild-type SpCas9 (default, backward compatible)
  SpCas9-NG  NGN   relaxed NG PAM
  SpG        NGN   engineered SpG
  VRQR       NGA   SpCas9-VRQR (Koblan et al. 2021 used ABEmax-VRQR for LMNA c.1824)
  SpRY       NRN   near-PAMless SpRY (R = A/G)

Genomic coordinates:
  Pass --chrom and --context_start (1-based genomic pos of context[0] on + strand)
  to emit ot_chrom / ot_start / ot_end for each candidate. The interval is the
  plus-strand span the 20nt protospacer occupies (identical for + and - designs;
  strand only labels orientation). Used downstream for coordinate-aware off-target
  exclusion (replaces the fragile subtract-one-exact-match assumption).

Modes:
  (default)          scan one PAM (via --cas_variant or --pam) and rank candidates
  --compare_variants scan every variant, print accessible-target counts, exit
  --all_variants     scan every variant, merge UNIQUE candidates into one ranked
                     table (each guide labeled by the most PAM-restrictive variant
                     that can reach it), then rank across all of them together

Pipeline (per candidate):
  pick strand -> scan PAM -> 20nt protospacer
  -> keep candidates placing target A in edit window (pos 4-8)
  -> B' efficiency prediction (physical features recomputed from protospacer)
  -> bystander risk (position-weighted empirical prior)
  -> off-target heuristic (offtarget_heuristic.py)
  -> 3-objective composite score -> Pareto flag -> Top-K

Composite: Score = w1*efficiency - w2*bystander_norm - w3*offtarget_risk
Presets: research 0.8/0.1/0.1, therapy 0.4/0.3/0.3, clinical 0.3/0.4/0.3
"""

import re
import json as _json
import pickle
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import RNA
from Bio.SeqUtils import MeltingTemp as mt

from offtarget_heuristic import offtarget_risk
from bystander_prior import bystander_risk, load_prior

# ------------------------------------------------------------------ config
PROTOSPACER_LEN = 20
EDIT_WINDOW = (4, 8)          # 1-based inclusive protospacer positions
BYSTANDER_NORM = 4.0
BYSTANDER_PRIOR = None

WEIGHT_PRESETS = {
    "research": {"w1": 0.8, "w2": 0.1, "w3": 0.1},
    "therapy":  {"w1": 0.4, "w2": 0.3, "w3": 0.3},
    "clinical": {"w1": 0.3, "w2": 0.4, "w3": 0.3},
}

# Cas variant -> PAM (IUPAC). SpCas9 NGG is default and identical to the old regex.
CAS_VARIANTS = {
    "SpCas9":    "NGG",
    "SpCas9-NG": "NGN",
    "SpG":       "NGN",
    "VRQR":      "NGA",
    "SpRY":      "NRN",
}

# Most-restrictive -> least-restrictive. Used by --all_variants so each guide is
# labeled by the most stringent variant that can reach it (fewer off-targets preferred).
VARIANT_ORDER = ["SpCas9", "VRQR", "SpCas9-NG", "SpG", "SpRY"]

_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}


# ------------------------------------------------------------------ conformal
def load_conformal_qhat(path, level="90"):
    """读 conformal_predict.py 产出的 q_hat；level 选 '80' 或 '90'。"""
    with open(path) as f:
        data = _json.load(f)
    return float(data["levels"][level]["q_hat"])


# ------------------------------------------------------------------ seq utils
def reverse_complement(seq):
    return "".join(_COMP.get(b, "N") for b in reversed(seq))


def pam_to_regex(pam):
    """Convert an IUPAC PAM string to a compiled overlapping-match regex."""
    pam = pam.upper()
    try:
        body = "".join(_IUPAC[b] for b in pam)
    except KeyError as e:
        raise ValueError(f"Invalid IUPAC base in PAM '{pam}': {e}")
    # overlapping lookahead so adjacent PAMs are all found; group(1) captures the PAM
    return re.compile(r"(?=(" + body + r"))")


from bprime_features import sequence_to_onehot, phys_from_grna

# ------------------------------------------------------------------ PAM scan
def _scan_strand_for_target(seq, target_idx, strand, pam_regex, seq_len):
    """
    Scan `seq` for the given PAM; keep candidates with target A in edit window.

    Records ctx_proto_start = 0-based index of the protospacer's leftmost base
    on the PLUS-strand context, so genomic coordinates can be computed uniformly
    for both strands. `seq_len` is len(context) (== len(seq)).
    """
    out = []
    for m in pam_regex.finditer(seq):
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

        # map protospacer span back to plus-strand context coordinates
        if strand == "+":
            ctx_proto_start = proto_start
        else:  # minus strand: seq is reverse_complement(context)
            ctx_proto_start = seq_len - proto_end

        out.append({
            "grna": protospacer,
            "pam": pam,
            "target_pos_in_proto": target_in_proto,
            "strand": strand,
            "ctx_proto_start": ctx_proto_start,
        })
    return out


def find_candidates(context, target_pos, pam_regex):
    """Auto-select the strand carrying the target A, then scan the PAM there."""
    context = context.upper()
    L = len(context)
    base = context[target_pos]
    if base == "A":
        return _scan_strand_for_target(context, target_pos, "+", pam_regex, L)
    if base == "T":
        rc = reverse_complement(context)
        return _scan_strand_for_target(rc, (L - 1) - target_pos, "-", pam_regex, L)
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
def score_candidates(candidates, model, weights, cas_variant,
                     chrom=None, context_start=None, conformal_q=None):
    if not candidates:
        return pd.DataFrame()
    effs = predict_efficiency(model, candidates)
    records = []
    for c, eff in zip(candidates, effs):
        by_count = bystander_a_count(c["grna"], c["target_pos_in_proto"])  # 保留:报告用
        br = bystander_risk(c["grna"], c["target_pos_in_proto"], BYSTANDER_PRIOR)
        by_norm = br["bystander_ratio"]   # 位置加权,已在[0,1)
        ot_score, ot_parts = offtarget_risk(c["grna"])
        composite = (weights["w1"] * float(eff)
                     - weights["w2"] * by_norm
                     - weights["w3"] * ot_score)

        # conformal prediction interval (optional)
        if conformal_q is not None:
            pred_low = round(float(np.clip(eff - conformal_q, 0.0, 1.0)), 3)
            pred_high = round(float(np.clip(eff + conformal_q, 0.0, 1.0)), 3)
        else:
            pred_low = None
            pred_high = None

        # genomic coordinates (1-based inclusive) of the plus-strand protospacer span
        if context_start is not None:
            ot_start = int(context_start) + int(c["ctx_proto_start"])
            ot_end = ot_start + PROTOSPACER_LEN - 1
        else:
            ot_start = None
            ot_end = None

        records.append({
            "grna": c["grna"],
            "pam": c["pam"],
            "cas_variant": cas_variant,
            "strand": c["strand"],
            "target_pos": c["target_pos_in_proto"],
            "pred_efficiency": float(eff),
            "pred_eff_low": pred_low,
            "pred_eff_high": pred_high,
            "bystander_A": by_count,
            "bystander_weighted": float(by_norm),
            "offtarget_risk": ot_score,
            "ot_seed": ot_parts["seed"],
            "ot_gc": ot_parts["gc"],
            "ot_homopolymer": ot_parts["homopolymer"],
            "ot_complexity": ot_parts["complexity"],
            "composite_score": float(composite),
            "ot_chrom": chrom,
            "ot_start": ot_start,
            "ot_end": ot_end,
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
    by = df["bystander_weighted"].values
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


# ------------------------------------------------------------------ variant comparison
def compare_variants(context, target_pos, model):
    """Scan every Cas variant; report accessible-candidate counts and best efficiency."""
    rows = []
    for name, pam in CAS_VARIANTS.items():
        rx = pam_to_regex(pam)
        cands = find_candidates(context, target_pos, rx)
        if cands:
            effs = predict_efficiency(model, cands)
            best = round(float(np.max(effs)), 3)
        else:
            best = None
        rows.append({
            "cas_variant": name,
            "pam": pam,
            "n_candidates": len(cands),
            "best_pred_eff": best,
        })
    return pd.DataFrame(rows)


def merge_all_variants(context, target_pos):
    """
    Scan every variant; return UNIQUE candidates keyed by (grna, strand),
    each labeled with the most PAM-restrictive variant that can reach it
    (VARIANT_ORDER: most -> least restrictive). Fewer off-targets preferred.
    """
    seen = {}
    for name in VARIANT_ORDER:
        rx = pam_to_regex(CAS_VARIANTS[name])
        for c in find_candidates(context, target_pos, rx):
            key = (c["grna"], c["strand"], c["ctx_proto_start"])
            if key not in seen:
                c2 = dict(c)
                c2["_variant"] = name
                seen[key] = c2
    return list(seen.values())


# ------------------------------------------------------------------ reporting
def print_report(df, scenario, weights, top_k, target_base, strand_used,
                 cas_variant, pam, conformal_level=None):
    print("\n" + "=" * 82)
    print(f"=== ABE8e candidate sgRNA ranking | scenario: {scenario} "
          f"(w1={weights['w1']} w2={weights['w2']} w3={weights['w3']}) ===")
    print(f"=== Cas variant: {cas_variant} (PAM {pam}) | "
          f"target base on plus strand: '{target_base}'  ->  "
          f"designing on {strand_used} strand | model: B' (seq+phys, no energy) ===")
    print("=" * 82)
    if df.empty:
        print(f"No candidates found (no {pam} PAM places the target A in window 4-8).")
        print("Try a PAM-relaxed variant: --cas_variant VRQR (NGA) or SpRY (NRN),")
        print("or run --compare_variants to see which variant unlocks this target.")
        return
    show = ["grna", "pam", "cas_variant", "strand", "target_pos", "pred_efficiency"]
    if conformal_level is not None and df["pred_eff_low"].notna().any():
        show += ["pred_eff_low", "pred_eff_high"]
    show += ["bystander_A", "bystander_weighted", "offtarget_risk", "composite_score"]
    if df["ot_start"].notna().any():
        show += ["ot_chrom", "ot_start", "ot_end"]
    if "pareto_optimal" in df.columns:
        show.append("pareto_optimal")
    print(df.head(top_k)[show].to_string())
    print("\n--- notes ---")
    print("model           : B' design model (80 one-hot + 3 physical, no energy)")
    print("                  physical features recomputed from protospacer (self-consistent)")
    print("cas_variant     : SpCas9 NGG / SpCas9-NG,SpG NGN / VRQR NGA / SpRY NRN")
    print("strand          : + = target A on plus strand, - = on minus strand")
    print("pred_efficiency : B' output (0-1, edited fraction); same-source held-out Test Spearman ~0.67")
    if conformal_level is not None and df["pred_eff_low"].notna().any():
        print(f"pred_eff_low/high: split-conformal {conformal_level}% prediction interval "
              f"(clip to [0,1]); calibrated on val")
    print("target_pos      : target A position in protospacer (configured candidate-generation window 4-8)")
    print("bystander_A     : extra A's in window excl. target (rule-based)")
    print("offtarget_risk  : heuristic intrinsic specificity risk (0-1, higher=worse)")
    print("composite_score : w1*eff - w2*bystander_norm - w3*offtarget_risk")
    if df["ot_start"].notna().any():
        print("ot_chrom/start/end : plus-strand genomic span of the 20nt protospacer (1-based)")
    print("pareto_optimal  : non-dominated across (eff up, bystander down, ot down)")
    print("\n[!] offtarget_risk is a sequence heuristic, NOT genome alignment.")
    print("    Real off-target spectrum needs Cas-OFFinder/CRISPOR + wet-lab (GUIDE-seq).")
    print("    Cas9-independent (deaminase) off-target is not captured here.")
    print("[!] PAM-relaxed variants (SpRY/NG) unlock more targets but usually have")
    print("    MORE genomic off-targets -> genome-wide Cas-OFFinder screening is essential.")


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(
        description="ABE8e candidate sgRNA design (dual-strand, multi-objective, B' model, PAM-variant aware)."
    )
    parser.add_argument("--context", required=True,
                        help="Plus-strand sequence containing the target base.")
    parser.add_argument("--target_pos", type=int, required=True,
                        help="0-based index of the target base within --context.")
    parser.add_argument("--model", default=str(ROOT / "models/xgboost_design_v2.pkl"),
                        help="B' efficiency model (default: models/xgboost_design_v2.pkl).")
    parser.add_argument("--scenario", choices=list(WEIGHT_PRESETS.keys()),
                        default="therapy")
    parser.add_argument("--cas_variant", choices=list(CAS_VARIANTS.keys()),
                        default="SpCas9",
                        help="Cas variant preset (default SpCas9 / NGG).")
    parser.add_argument("--pam", default=None,
                        help="Custom IUPAC PAM (overrides --cas_variant), e.g. NGN, NGA, NRN, NNN.")
    parser.add_argument("--compare_variants", action="store_true",
                        help="Scan all Cas variants and print accessible-target counts, then exit.")
    parser.add_argument("--all_variants", action="store_true",
                        help="Scan ALL Cas variants, merge unique candidates into one ranked table "
                             "(labeled with the most PAM-restrictive variant that reaches each).")
    parser.add_argument("--chrom", default=None,
                        help="Chromosome for genomic coordinate output, e.g. chr1.")
    parser.add_argument("--context_start", type=int, default=None,
                        help="1-based genomic position of context[0] on + strand "
                             "(e.g. window_start_1based from extract_context.py).")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--conformal", action="store_true",
                        help="Attach conformal prediction interval columns (pred_eff_low/high).")
    parser.add_argument("--conformal_json",
                        default=str(ROOT / "results/conformal/conformal_qhat.json"),
                        help="Path to conformal_predict.py output json.")
    parser.add_argument("--conformal_level", choices=["80", "90"], default="90",
                        help="Prediction-interval nominal level (default 90).")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--prior",
        default=str(ROOT / "results/bystander/empirical_prior.json"),
        help="Empirical position-prior JSON; no automatic fallback."
    )
    args = parser.parse_args()

    global BYSTANDER_PRIOR
    BYSTANDER_PRIOR = load_prior(args.prior)

    if args.top_k < 1:
        parser.error("--top_k must be positive.")
    if (args.chrom is None) != (args.context_start is None):
        parser.error("--chrom and --context_start must be provided together.")
    if args.context_start is not None and args.context_start < 1:
        parser.error("--context_start must be a positive 1-based coordinate.")
    if args.compare_variants and args.all_variants:
        parser.error("--compare_variants and --all_variants are mutually exclusive.")
    if args.pam is not None and not args.pam.strip():
        parser.error("--pam cannot be empty.")
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    context = args.context.upper().strip()
    if not context or set(context) - set("ACGT"):
        raise ValueError("This release requires context containing only A/C/G/T.")
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

    conformal_q = None
    conformal_level = None
    if args.conformal:
        conformal_q = load_conformal_qhat(args.conformal_json, args.conformal_level)
        conformal_level = args.conformal_level
        print(f"Conformal {args.conformal_level}% PI enabled, half-width q = {conformal_q:.4f}")

    # --- variant comparison mode (report figure data) ---
    if args.compare_variants:
        print(f"\nTarget base '{target_base}' at index {args.target_pos} "
              f"-> scanning {strand_used} strand (context length {len(context)}).")
        print("\n=== PAM-variant accessibility comparison ===")
        cmp_df = compare_variants(context, args.target_pos, model)
        print(cmp_df.to_string(index=False))
        print("\nInterpretation: variants with n_candidates > 0 can access this target.")
        print("Accessibility counts reflect the configured PAM and editing-window rules.")
        print("Cross-Cas efficiency calibration has not been established.")
        if args.out_csv:
            cmp_df.to_csv(args.out_csv, index=False)
            print(f"\nSaved variant comparison -> {args.out_csv}")
        return

    # --- all-variants mode: merge unique candidates across every variant ---
    if args.all_variants:
        merged = merge_all_variants(context, args.target_pos)
        print(f"\nTarget base '{target_base}' at index {args.target_pos} "
              f"-> scanning {strand_used} strand (context length {len(context)}).")
        print(f"Merged {len(merged)} unique candidate(s) across all PAM variants.")
        if not merged:
            print("No candidates from any variant.")
            return
        # score each candidate labeled with its most-restrictive reachable variant
        frames = []
        for c in merged:
            df_c = score_candidates([c], model, weights, c["_variant"],
                                    chrom=args.chrom, context_start=args.context_start,
                                    conformal_q=conformal_q)
            frames.append(df_c)
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        df = pareto_front(df)
        print_report(df, args.scenario, weights, args.top_k, target_base,
                     strand_used, "ALL", "mixed", conformal_level=conformal_level)
        if args.out_csv:
            df.to_csv(args.out_csv)
            print(f"\nSaved merged candidate table -> {args.out_csv}")
        return

    # --- resolve PAM ---
    if args.pam is not None:
        pam = args.pam.upper()
        cas_variant = f"custom({pam})"
    else:
        pam = CAS_VARIANTS[args.cas_variant]
        cas_variant = args.cas_variant
    pam_regex = pam_to_regex(pam)

    print(f"Target base '{target_base}' at index {args.target_pos} "
          f"-> scanning {strand_used} strand (context length {len(context)}) "
          f"with {cas_variant} PAM {pam}...")
    candidates = find_candidates(context, args.target_pos, pam_regex)
    print(f"Found {len(candidates)} candidate(s) with target A in window 4-8.")

    df = score_candidates(candidates, model, weights, cas_variant,
                          chrom=args.chrom, context_start=args.context_start,
                          conformal_q=conformal_q)
    if not df.empty:
        df = pareto_front(df)
    print_report(df, args.scenario, weights, args.top_k, target_base,
                 strand_used, cas_variant, pam, conformal_level=conformal_level)

    if args.out_csv and not df.empty:
        df.to_csv(args.out_csv)
        print(f"\nSaved full ranked table -> {args.out_csv}")


if __name__ == "__main__":
    main()