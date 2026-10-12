"""
Prepare clean training dataset from Study 3
- Filter study_id = 3
- Use efficiency_full_grna_reported as target
- Remove rows with missing energy features
- Remove rows with missing target
- Split train/val/test with sequence-level grouping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]

(ROOT / "data/processed").mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(ROOT / "data/raw/efficiency_ABE8e_raw.csv")

print("Original shape:", df.shape)

# Filter Study 3
study3 = df[df["study_id"] == 3].copy()
print("Study 3 shape:", study3.shape)

# Use reported efficiency as target
study3 = study3[study3["efficiency_full_grna_reported"].notna()]
print("After removing missing target:", study3.shape)

# Remove rows with missing energy features
energy_cols = [f"energy_{i}" for i in range(1, 25)]
study3 = study3.dropna(subset=energy_cols)
print("After removing missing energy:", study3.shape)

# Check remaining missing values
print("\nRemaining missing values:")
print(study3.isnull().sum()[study3.isnull().sum() > 0])

# === Split strategy: prevent sequence leakage ===
# Group by full_context_sequence to ensure same target doesn't appear in train and test
unique_seqs = study3["full_context_sequence"].unique()
print(f"\nUnique target sequences: {len(unique_seqs)}")

# Split sequences first
train_seqs, temp_seqs = train_test_split(
    unique_seqs, 
    test_size=0.3, 
    random_state=42
)

val_seqs, test_seqs = train_test_split(
    temp_seqs,
    test_size=0.5,
    random_state=42
)

# Map back to data
train = study3[study3["full_context_sequence"].isin(train_seqs)]
val = study3[study3["full_context_sequence"].isin(val_seqs)]
test = study3[study3["full_context_sequence"].isin(test_seqs)]

print("\n=== Data Split ===")
print(f"Train: {train.shape[0]} samples ({len(train_seqs)} unique sequences)")
print(f"Val:   {val.shape[0]} samples ({len(val_seqs)} unique sequences)")
print(f"Test:  {test.shape[0]} samples ({len(test_seqs)} unique sequences)")

# Target distribution check
print("\n=== Target Distribution ===")
for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
    target = subset["efficiency_full_grna_reported"]
    print(f"{name}: mean={target.mean():.3f}, std={target.std():.3f}, "
          f"median={target.median():.3f}")

# Save
train.to_csv(ROOT / "data/processed/train.csv", index=False)
val.to_csv(ROOT / "data/processed/val.csv", index=False)
test.to_csv(ROOT / "data/processed/test.csv", index=False)

print("\n=== Saved to data/processed/ ===")

# === Prepare Study 4 as external test (different label) ===
study4 = df[df["study_id"] == 4].copy()
study4.to_csv(ROOT / "data/processed/external_test_study4.csv", index=False)
print(f"\nExternal test (Study 4): {study4.shape[0]} samples")
print("Note: Uses efficiency_full_grna_calculated (different from training target)")
# Preserve the actual historical split assignment, without re-sorting sequences.
split_dir = ROOT / "data/splits"
split_dir.mkdir(parents=True, exist_ok=True)
split_rows = []
for split_name, frame in [("train", train), ("val", val), ("test", test)]:
    for sequence in frame["full_context_sequence"].drop_duplicates():
        if not isinstance(sequence, str):
            raise ValueError("Missing or non-string full_context_sequence.")
        split_rows.append({
            "context_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            "split": split_name,
        })
pd.DataFrame(split_rows).to_csv(
    split_dir / "study3_split_manifest.csv", index=False
)