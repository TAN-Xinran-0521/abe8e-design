# Research heuristic; not a calibrated off-target probability.
"""
Heuristic sgRNA specificity / off-target-propensity scoring.

NOT genome alignment. Produces an INTRINSIC off-target risk score from
sequence features known to affect Cas9/ABE guide-directed specificity.
Used as a penalty/filter term in multi-objective ranking.

Limitations (state these in the report):
  - Real off-target spectrum needs whole-genome alignment (Cas-OFFinder / CRISPOR).
  - Cas9-independent (deaminase) off-target is NOT captured here.

Score in [0,1]: higher = higher off-target propensity (worse).
"""

import numpy as np
from collections import Counter


def gc_content(seq):
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def max_homopolymer(seq):
    """Length of the longest run of identical bases."""
    if not seq:
        return 0
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def dinuc_entropy(seq):
    """Normalized dinucleotide Shannon entropy (0-1). Low = repetitive."""
    if len(seq) < 2:
        return 0.0
    dinucs = [seq[i:i + 2] for i in range(len(seq) - 1)]
    counts = Counter(dinucs)
    total = len(dinucs)
    ent = -sum((c / total) * np.log2(c / total) for c in counts.values())
    return ent / 4.0  # log2(16) = 4


# ---- per-factor risk (0 = safe, 1 = risky) ----

def gc_risk(seq):
    """Ideal 40-60%. Ramp to 1.0 outside 30-70%."""
    gc = gc_content(seq)
    if 0.4 <= gc <= 0.6:
        return 0.0
    if gc < 0.3 or gc > 0.7:
        return 1.0
    return (0.4 - gc) / 0.1 if gc < 0.4 else (gc - 0.6) / 0.1


def homopolymer_risk(seq):
    """Runs of 4+ identical bases raise risk (synthesis + specificity)."""
    run = max_homopolymer(seq)
    return 0.0 if run <= 3 else min(1.0, (run - 3) / 3.0)


def complexity_risk(seq):
    """Low sequence complexity -> matches more genomic sites -> higher risk."""
    return 1.0 - dinuc_entropy(seq)


def seed_risk(seq, seed_len=10):
    """PAM-proximal seed (3' end for SpCas9): specificity-critical region."""
    seed = seq[-seed_len:]
    return 0.5 * complexity_risk(seed) + 0.5 * homopolymer_risk(seed)


def offtarget_risk(seq, weights=None):
    """
    Returns (risk_score in [0,1], per-factor breakdown dict).
    Seed weighted highest — PAM-proximal specificity dominates off-target.
    """
    seq = seq.upper().strip()
    if len(seq) != 20 or set(seq) - set("ACGT"):
        raise ValueError("Expected a 20nt A/C/G/T guide.")
    w = (
        {"seed": 0.4, "gc": 0.2, "homopolymer": 0.2, "complexity": 0.2}
        if weights is None else dict(weights)
    )
    if set(w) != {"seed", "gc", "homopolymer", "complexity"}:
        raise ValueError("Unexpected heuristic weight keys.")
    if any(not np.isfinite(v) or v < 0 for v in w.values()):
        raise ValueError("Weights must be finite and nonnegative.")
    if not np.isclose(sum(w.values()), 1.0):
        raise ValueError("Heuristic weights must sum to 1.")
    parts = {
        "seed": seed_risk(seq),
        "gc": gc_risk(seq),
        "homopolymer": homopolymer_risk(seq),
        "complexity": complexity_risk(seq),
    }
    score = sum(w[k] * parts[k] for k in w)
    return float(score), {k: float(v) for k, v in parts.items()}


if __name__ == "__main__":
    demos = [
        "GATCAATCGATCGATCGATC",
        "AAAAAAAAAAAAAAAAAAAA",
        "GCGCGCGCGCGCGCGCGCGC",
        "ATGCTAGCATCGTAGCTAGC",
    ]
    print(f"{'sgRNA':<24}{'risk':>6}   breakdown")
    print("-" * 70)
    for s in demos:
        score, parts = offtarget_risk(s)
        print(f"{s:<24}{score:>6}   {parts}")