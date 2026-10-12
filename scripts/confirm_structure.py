"""Check bystander data structure"""
import pandas as pd

bys = pd.read_csv("data/raw/bystander_ABE8e_raw.csv")

print("Shape:", bys.shape)
print("\nColumns:")
print(bys.columns.tolist())

print("\n=== Study distribution ===")
print(bys["study_id"].value_counts())

# Find position columns (逐位点编辑频率)
position_cols = [c for c in bys.columns if c.startswith("position_")]
print(f"\nPosition columns: {len(position_cols)}")
print(position_cols[:10])  # 显示前10个