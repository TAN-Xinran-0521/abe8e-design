"""
XGBoost baseline for ABE8e efficiency prediction
- One-hot encoding for gRNA sequence
- Energy features + melting temperature
- Target: efficiency_full_grna_reported
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb
import pickle
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === Feature engineering ===
def sequence_to_onehot(seq):
    """Convert DNA sequence to one-hot encoding"""
    mapping = {'A': [1,0,0,0], 'T': [0,1,0,0], 'G': [0,0,1,0], 'C': [0,0,0,1]}
    encoded = []
    for nucleotide in seq:
        encoded.extend(mapping.get(nucleotide, [0,0,0,0]))
    return encoded

def extract_features(df):
    """Extract all features for XGBoost"""
    features = []
    
    # One-hot encode gRNA (20nt × 4 = 80 dims)
    grna_onehot = df["grna"].apply(sequence_to_onehot).tolist()
    grna_onehot = np.array(grna_onehot)
    
    # Energy features (24 dims)
    energy_cols = [f"energy_{i}" for i in range(1, 25)]
    energy_features = df[energy_cols].values
    
    # Physical features (3 dims)
    physical_features = df[["free_energy", "melt_temperature_grna", "melt_temperature_target"]].values
    
    # Concatenate
    features = np.hstack([grna_onehot, energy_features, physical_features])
    
    return features

# === Load data ===
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

print("Extracting features...")
X_train = extract_features(train)
y_train = train["efficiency_full_grna_reported"].values

X_val = extract_features(val)
y_val = val["efficiency_full_grna_reported"].values

X_test = extract_features(test)
y_test = test["efficiency_full_grna_reported"].values

print(f"Feature dimensions: {X_train.shape[1]}")

# === Train XGBoost ===
print("\nTraining XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Save model
with open("models/xgboost_baseline.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to models/xgboost_baseline.pkl")

# === Evaluate ===
def evaluate(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman = spearmanr(y_true, y_pred)[0]
    pearson = pearsonr(y_true, y_pred)[0]
    
    print(f"\n=== {dataset_name} Results ===")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Pearson:  {pearson:.4f}")
    
    return {
        "dataset": dataset_name,
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "pearson": pearson
    }

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluate all splits
results = []
results.append(evaluate(y_train, y_train_pred, "Train"))
results.append(evaluate(y_val, y_val_pred, "Validation"))
results.append(evaluate(y_test, y_test_pred, "Test"))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results/xgboost_baseline_results.csv", index=False)
print("\nResults saved to results/xgboost_baseline_results.csv")