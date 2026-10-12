"""Shared B-prime feature recipe.

Order: 80 positional one-hot features (A,T,G,C), then MFE, Tm, Tm.
The duplicate Tm column is retained for compatibility with the existing
83-feature trained model. Do not remove it without retraining.
"""
import numpy as np
import RNA
from Bio.SeqUtils import MeltingTemp as mt

FEATURE_RECIPE_VERSION = "bprime_83_v1"

def validate_protospacer(seq):
    if not isinstance(seq, str):
        raise ValueError("Protospacer must be a string.")
    seq = seq.strip().upper()
    if len(seq) != 20 or set(seq) - set("ACGT"):
        raise ValueError("Protospacer must contain exactly 20 A/C/G/T bases.")
    return seq

def sequence_to_onehot(seq):
    seq = validate_protospacer(seq)
    mapping = {
        "A": [1, 0, 0, 0], "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0], "C": [0, 0, 0, 1],
    }
    return [value for base in seq for value in mapping[base]]

def phys_from_grna(seq):
    seq = validate_protospacer(seq)
    _, mfe = RNA.fold_compound(seq.replace("T", "U")).mfe()
    tm = float(mt.Tm_NN(seq))
    return [float(mfe), tm, tm]

def build_X(df):
    rows = [
        sequence_to_onehot(seq) + phys_from_grna(seq)
        for seq in df["grna"]
    ]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 83)