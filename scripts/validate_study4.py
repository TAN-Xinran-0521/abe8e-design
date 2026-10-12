"""
External validation on Study 4 (cross-study generalization)
Note: Study 4 uses efficiency_full_grna_calculated (different computation)
We evaluate rank correlation (Spearman) instead of absolute values
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr, pearsonr
from train_baseline import extract_features
import matplotlib.pyplot as plt

# Load model
with open("models/xgboost_baseline.pkl", "rb") as f:
    model = pickle.load(f)

# Load Study 4
study4 = pd.read_csv("data/processed/external_test_study4.csv")

print("=== Study 4 Info ===")
print("Shape:", study4.shape)
print("Sequence length distribution:")
print(study4["sequence"].str.len().value_counts())
print("\nTarget column: efficiency_full_grna_calculated")
print(f"Missing values: {study4['efficiency_full_grna_calculated'].isna().sum()}")

# Study 4 has different sequence format (20/20 vs 40/79 in Study 3)
# We need to handle this

# Check if Study 4 has energy features
energy_cols = [f"energy_{i}" for i in range(1, 25)]
missing_energy = study4[energy_cols].isna().sum().sum()
print(f"\nMissing energy values: {missing_energy}")

if missing_energy == 0:
    # Extract features normally
    X_study4 = extract_features(study4)
    y_study4 = study4["efficiency_full_grna_calculated"].values
    
    print("\n=== Predicting on Study 4 ===")
    y_pred = model.predict(X_study4)
    
    # Evaluate rank correlation (most important for cross-study)
    spearman = spearmanr(y_study4, y_pred)[0]
    pearson = pearsonr(y_study4, y_pred)[0]
    
    print("\n=== Cross-Study Validation Results ===")
    print(f"Spearman correlation: {spearman:.4f}")
    print(f"Pearson correlation:  {pearson:.4f}")
    print("\nNote: Absolute efficiency values may differ due to different")
    print("      calculation methods. Rank correlation is more reliable.")
    
    # Save predictions
    study4["predicted_efficiency"] = y_pred
    study4.to_csv("results/study4_predictions.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(y_study4, y_pred, alpha=0.5, s=10)
    plt.xlabel("Observed (calculated)")
    plt.ylabel("Predicted (from Study 3 model)")
    plt.title(f"Cross-Study Validation\nSpearman: {spearman:.3f}")
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/study4_validation.png", dpi=150)
    print("\nScatter plot saved to results/study4_validation.png")
    
else:
    print("\nWARNING: Study 4 has missing energy features.")
    print("Cannot perform direct prediction without feature imputation.")
    print("Skipping Study 4 validation for now.")