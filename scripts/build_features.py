"""
Build and cache feature matrices from processed CSV splits.
Saves X/y as .npz so downstream scripts (baseline, transformer, SHAP)
load identical features without recomputation.
"""

import numpy as np
import pandas as pd
import os

os.makedirs("data/features", exist_ok=True)

def sequence_to_onehot(seq):
    mapping = {'A':[1,0,0,0],'T':[0,1,0,0],'G':[0,0,1,0],'C':[0,0,0,1]}
    out = []
    for n in seq:
        out.extend(mapping.get(n, [0,0,0,0]))
    return out

ENERGY_COLS = [f"energy_{i}" for i in range(1, 25)]
PHYS_COLS = ["free_energy", "melt_temperature_grna", "melt_temperature_target"]
TARGET = "efficiency_full_grna_reported"

def extract_features(df):
    grna = np.array(df["grna"].apply(sequence_to_onehot).tolist())
    energy = df[ENERGY_COLS].values
    phys = df[PHYS_COLS].values
    return np.hstack([grna, energy, phys])

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/processed/{split}.csv")
    X = extract_features(df)
    y = df[TARGET].values
    np.savez(f"data/features/{split}.npz", X=X, y=y)
    print(f"{split}: X={X.shape}, y={y.shape}")

print("Saved to data/features/")