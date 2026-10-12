"""
Validate BE-dataHIVE ABE8e dataset

Checks:
- file integrity
- schema
- sequence information
- label distribution
- missing values
- biological metadata
- potential leakage

"""

import os
import pandas as pd
import numpy as np


# =====================================================
# Config
# =====================================================

DATA_FILE = (
    "data/raw/"
    "efficiency_ABE8e_raw.csv"
)


REPORT_FILE = (
    "data/raw/"
    "efficiency_validation_report.txt"
)



# =====================================================
# Helper
# =====================================================


def section(title):

    print("\n")
    print("="*70)
    print(title)
    print("="*70)



def write_report(text):

    with open(
        REPORT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(text+"\n")



def show(text):

    print(text)
    write_report(str(text))



# =====================================================
# Load
# =====================================================


if os.path.exists(REPORT_FILE):

    os.remove(REPORT_FILE)



section("1. Loading dataset")


if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        DATA_FILE
    )


df = pd.read_csv(
    DATA_FILE
)


show(
    f"Dataset path: {DATA_FILE}"
)

show(
    f"Shape: {df.shape}"
)



# =====================================================
# Columns
# =====================================================


section("2. Column inspection")


show(
    "Columns:"
)

for c in df.columns:

    show(
        f" - {c}"
    )



# =====================================================
# Required fields
# =====================================================


section("3. Required column check")


required = [

    # sequence
    "grna",
    "pam_sequence",
    "sequence",
    "full_context_sequence",

    # metadata
    "cell",
    "base_editor",

    # physics features
    "free_energy",
    "melt_temperature_grna",
    "melt_temperature_target",

    # label
    "efficiency_full_grna_calculated"

]


missing = [

    c for c in required
    if c not in df.columns

]


if len(missing)==0:

    show(
        "PASS: all required columns exist"
    )

else:

    show(
        f"WARNING missing columns: {missing}"
    )



# =====================================================
# Base editor
# =====================================================


section("4. Base editor check")


show(
    df["base_editor"]
    .value_counts()
    .to_string()
)


if len(df["base_editor"].unique())==1:

    show(
        "PASS: single editor dataset"
    )



# =====================================================
# Sequence length
# =====================================================


section("5. Sequence length check")


for col in [

    "grna",
    "sequence",
    "full_context_sequence"

]:

    if col in df.columns:


        lengths = (
            df[col]
            .astype(str)
            .str.len()
            .value_counts()
            .head(10)
        )


        show(
            f"\n{col} length distribution:"
        )


        show(
            lengths.to_string()
        )



# =====================================================
# Label statistics
# =====================================================


section("6. Efficiency label statistics")


target = (
    "efficiency_full_grna_calculated"
)


if target in df.columns:


    desc = (
        df[target]
        .describe()
    )


    show(
        desc.to_string()
    )


    show(
        "\nMissing label:"
    )


    show(
        str(
            df[target]
            .isna()
            .sum()
        )
    )


    show(
        "\nZero efficiency:"
    )


    show(
        str(
            (
                df[target]==0
            )
            .sum()
        )
    )



# =====================================================
# Missing values
# =====================================================


section("7. Missing value analysis")


missing_rate = (
    df.isna()
    .mean()
    .sort_values(
        ascending=False
    )
)


show(
    missing_rate
    .head(20)
    .to_string()
)



# =====================================================
# Metadata
# =====================================================


section("8. Biological metadata")


for col in [

    "cell",
    "study_id"

]:

    if col in df.columns:


        show(
            f"\n{col} distribution:"
        )


        show(
            df[col]
            .value_counts()
            .head(20)
            .to_string()
        )



# =====================================================
# Duplicate check
# =====================================================


section("9. Duplicate check")


dup = (
    df
    .duplicated()
    .sum()
)


show(
    f"Duplicated rows: {dup}"
)

# =====================================================
# Sequence duplicate
# =====================================================
section("10. Target sequence redundancy")
if "full_context_sequence" in df.columns:
    seq_dup = (
        df[
            "full_context_sequence"
        ]
        .duplicated()
        .sum()
    )
    show(
        f"Duplicated sequences: {seq_dup}"
    )

# =====================================================
# Feature availability
# =====================================================
section("11. Feature availability")

energy_cols = [
    c for c in df.columns
    if c.startswith("energy_")
]

show(
    f"Energy features: {len(energy_cols)}"
)
show(
    energy_cols
)

# =====================================================
# Leakage check
# ====================================================
section("12. Potential leakage check")


danger = [
    "edited_count",
    "total_count",
    "efficiency_full_grna_reported",
    "efficiency_full_grna_calculated"
]

for c in danger:
    if c in df.columns:
        show(
            f"WARNING: {c} exists"
        )
show(
"""
Recommended prediction target:
efficiency_full_grna_calculated

Do NOT use:
- edited_count
- total_count
- reported efficiency
as input features.
"""
)
# =====================================================
# Final
# =====================================================
section("VALIDATION COMPLETE")
show(
    "Validation report saved:"
)
show(
    REPORT_FILE
)