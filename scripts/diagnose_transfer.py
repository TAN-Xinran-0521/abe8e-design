"""
Diagnose cross-study transfer failure.
Train gRNA-sequence-only model, test on both Study 3 test and Study 4.
Isolates whether energy features are the cause of negative correlation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

def sequence_to_onehot(seq):
    mapping = {'A':[1,0,0,0],'T':[0,1,0,0],'G':[0,0,1,0],'C':[0,0,0,1]}
    out = []
    for n in seq:
        out.extend(mapping.get(n, [0,0,0,0]))
    return out

def grna_features(df):
    return np.array(df["grna"].apply(sequence_to_onehot).tolist())

# Load
train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")
study4 = pd.read_csv("data/processed/external_test_study4.csv")

X_train = grna_features(train)
y_train = train["efficiency_full_grna_reported"].values

X_test = grna_features(test)
y_test = test["efficiency_full_grna_reported"].values

X_s4 = grna_features(study4)
y_s4 = study4["efficiency_full_grna_calculated"].values

# Train gRNA-only model
model = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("=== gRNA-ONLY model (80 features) ===")
sp_test = spearmanr(y_test, model.predict(X_test))[0]
sp_s4   = spearmanr(y_s4,  model.predict(X_s4))[0]
print(f"Study 3 Test  Spearman: {sp_test:.4f}")
print(f"Study 4 (ext) Spearman: {sp_s4:.4f}")

# Also: check energy feature distribution shift
print("\n=== Energy feature distribution: Study3 train vs Study4 ===")
energy_cols = [f"energy_{i}" for i in range(1, 25)]
comp = pd.DataFrame({
    "study3_mean": train[energy_cols].mean(),
    "study4_mean": study4[energy_cols].mean(),
    "study3_std":  train[energy_cols].std(),
    "study4_std":  study4[energy_cols].std(),
})
print(comp.round(3).to_string())