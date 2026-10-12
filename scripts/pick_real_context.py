"""
Pick real ABE8e contexts from the held-out test set whose target A naturally
falls in the edit window (protospacer pos 4-8), and print ready-to-run
design_sgrna.py commands. Finds one plus-strand and one minus-strand example.
"""

import pandas as pd

TEST_CSV = "data/processed/test.csv"
CONTEXT_COL = "full_context_sequence"
GRNA_COL = "grna"
LABEL_COL = "efficiency_full_grna_reported"
WINDOW = (4, 8)  # 1-based inclusive

_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def revcomp(s):
    return "".join(_COMP.get(b, "N") for b in reversed(s))


def find_example(df, want_strand):
    """Return (context, target_pos, grna, real_eff) for the wanted strand."""
    for _, r in df.iterrows():
        context = str(r[CONTEXT_COL]).upper()
        grna = str(r[GRNA_COL]).upper()
        if len(grna) != 20:
            continue

        if want_strand == "+":
            idx = context.find(grna)
            if idx < 0:
                continue
            # first A in window 4-8 on the plus-strand protospacer
            for p in range(WINDOW[0], WINDOW[1] + 1):
                if grna[p - 1] == "A":
                    target_pos = idx + (p - 1)      # plus-strand context coord
                    if context[target_pos] == "A":  # sanity
                        return context, target_pos, grna, r[LABEL_COL]
        else:  # minus strand
            rc = revcomp(grna)
            idx = context.find(rc)                  # rc of grna appears on plus
            if idx < 0:
                continue
            # minus-strand protospacer pos p (1-based from its 5' end)
            # maps to plus-strand coord idx + (20 - p); that plus base is 'T'
            for p in range(WINDOW[0], WINDOW[1] + 1):
                if grna[p - 1] == "A":
                    target_pos = idx + (20 - p)
                    if 0 <= target_pos < len(context) and context[target_pos] == "T":
                        return context, target_pos, grna, r[LABEL_COL]
    return None


def main():
    df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df)} test rows.\n")

    for strand in ("+", "-"):
        res = find_example(df, strand)
        label = "PLUS strand" if strand == "+" else "MINUS strand"
        print("=" * 70)
        print(f"{label} example")
        print("=" * 70)
        if res is None:
            print("(none found)\n")
            continue
        context, target_pos, grna, real_eff = res
        print(f"real grna        : {grna}")
        print(f"real efficiency  : {real_eff:.3f}")
        print(f"context ({len(context)}bp) : {context}")
        print(f"target_pos       : {target_pos}  (context base = '{context[target_pos]}')")
        print("\nRun this command:")
        print(f"python scripts/design_sgrna.py --context {context} "
              f"--target_pos {target_pos} --scenario therapy --top_k 5 "
              f"--out_csv results/candidates_{'plus' if strand=='+' else 'minus'}.csv")
        print()


if __name__ == "__main__":
    main()