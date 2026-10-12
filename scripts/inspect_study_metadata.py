"""
Inspect Study 3 and Study 4 metadata to understand transfer failure
Check: original publication, editor, cell line, label source
"""

import pandas as pd

# Load both datasets
study3 = pd.read_csv("data/processed/train.csv")  # 用train代表study3
study4 = pd.read_csv("data/processed/external_test_study4.csv")

print("="*70)
print("STUDY 3 METADATA")
print("="*70)

# Basic info
print(f"Sample count: {len(study3)}")
print(f"Study ID: {study3['study_id'].unique()}")

# Cell and editor
print(f"\nCell line(s): {study3['cell'].unique()}")
print(f"Base editor(s): {study3['base_editor'].unique()}")

# Label availability and distribution
print(f"\nLabel column: efficiency_full_grna_reported")
print(f"  Available: {study3['efficiency_full_grna_reported'].notna().sum()}")
print(f"  Missing: {study3['efficiency_full_grna_reported'].isna().sum()}")
print(f"\nLabel distribution:")
print(study3['efficiency_full_grna_reported'].describe())

# Sequence format
print(f"\nSequence format:")
print(f"  grna length: {study3['grna'].str.len().unique()}")
print(f"  full_context_sequence length: {study3['full_context_sequence'].str.len().unique()}")

print("\n" + "="*70)
print("STUDY 4 METADATA")
print("="*70)

print(f"Sample count: {len(study4)}")
print(f"Study ID: {study4['study_id'].unique()}")

print(f"\nCell line(s): {study4['cell'].unique()}")
print(f"Base editor(s): {study4['base_editor'].unique()}")

print(f"\nLabel column: efficiency_full_grna_calculated")
print(f"  Available: {study4['efficiency_full_grna_calculated'].notna().sum()}")
print(f"  Missing: {study4['efficiency_full_grna_calculated'].isna().sum()}")
print(f"\nLabel distribution:")
print(study4['efficiency_full_grna_calculated'].describe())

print(f"\nSequence format:")
print(f"  grna length: {study4['grna'].str.len().unique()}")
print(f"  full_context_sequence length: {study4['full_context_sequence'].str.len().unique()}")

# Check if there are any publication/reference columns
print("\n" + "="*70)
print("SEARCHING FOR PUBLICATION METADATA")
print("="*70)

all_cols_study3 = study3.columns.tolist()
all_cols_study4 = study4.columns.tolist()

pub_keywords = ['study', 'publication', 'paper', 'reference', 'author', 'pmid', 'doi']

print("\nStudy 3 columns containing metadata keywords:")
for col in all_cols_study3:
    if any(kw in col.lower() for kw in pub_keywords):
        print(f"  - {col}: {study3[col].unique()[:3]}")

print("\nStudy 4 columns containing metadata keywords:")
for col in all_cols_study4:
    if any(kw in col.lower() for kw in pub_keywords):
        print(f"  - {col}: {study4[col].unique()[:3]}")

# Label comparison side-by-side
print("\n" + "="*70)
print("LABEL COMPARISON")
print("="*70)

comp = pd.DataFrame({
    'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
    'Study3_reported': [
        study3['efficiency_full_grna_reported'].notna().sum(),
        study3['efficiency_full_grna_reported'].mean(),
        study3['efficiency_full_grna_reported'].std(),
        study3['efficiency_full_grna_reported'].min(),
        study3['efficiency_full_grna_reported'].quantile(0.25),
        study3['efficiency_full_grna_reported'].median(),
        study3['efficiency_full_grna_reported'].quantile(0.75),
        study3['efficiency_full_grna_reported'].max()
    ],
    'Study4_calculated': [
        study4['efficiency_full_grna_calculated'].notna().sum(),
        study4['efficiency_full_grna_calculated'].mean(),
        study4['efficiency_full_grna_calculated'].std(),
        study4['efficiency_full_grna_calculated'].min(),
        study4['efficiency_full_grna_calculated'].quantile(0.25),
        study4['efficiency_full_grna_calculated'].median(),
        study4['efficiency_full_grna_calculated'].quantile(0.75),
        study4['efficiency_full_grna_calculated'].max()
    ]
})

print(comp.round(4).to_string(index=False))

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("If Study 3 and Study 4 differ in:")
print("  - Editor type → context-dependent performance expected")
print("  - Cell line → biological transfer failure expected")
print("  - Label source (reported vs calculated) → measurement inconsistency")
print("  - Sequence format → feature mismatch")
print("\nThese differences explain the negative Spearman correlation.")