"""
Download required BE-dataHIVE data for ABE8e optimization project.

Features:
- Direct API access (avoid wrapper auto-download)
- Only ABE8e data
- Manual pagination
- Resume from checkpoint
- Handle 429/504 errors
- Save only required columns
"""

import os
import json
import time
import pandas as pd

from be_datahive import be_datahive

# =====================================================
# Configuration
# =====================================================
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

EFF_OUTPUT = os.path.join(
    RAW_DIR,
    "efficiency_ABE8e_raw.csv"
)

BYS_OUTPUT = os.path.join(
    RAW_DIR,
    "bystander_ABE8e_raw.csv"
)

CHECKPOINT = os.path.join(
    RAW_DIR,
    "download_checkpoint.json"
)

# API parameter
BASE_EDITOR = "ABE8e"
LIMIT = 500          # stable value after testing
SLEEP_TIME = 8       # seconds between requests
MAX_RETRY = 10
api = be_datahive()

# =====================================================
# Required columns
# =====================================================
EFF_COLUMNS = [
    "grna",
    "pam_sequence",
    "sequence",
    "full_context_sequence",
    "full_context_sequence_padded",
    "protospace_position",
    "pam_index",
    "cell",
    "base_editor",
    "energy_1",
    "energy_2",
    "energy_3",
    "energy_4",
    "energy_5",
    "energy_6",
    "energy_7",
    "energy_8",
    "energy_9",
    "energy_10",
    "energy_11",
    "energy_12",
    "energy_13",
    "energy_14",
    "energy_15",
    "energy_16",
    "energy_17",
    "energy_18",
    "energy_19",
    "energy_20",
    "energy_21",
    "energy_22",
    "energy_23",
    "energy_24",
    "free_energy",
    "melt_temperature_grna",
    "melt_temperature_target",
    "efficiency_full_grna_calculated",
    "efficiency_full_grna_reported",
    "study_id"
]

# bystander:
# Position_x columns are added dynamically
BYS_BASE_COLUMNS = [
    "grna",
    "pam_sequence",
    "sequence",
    "full_context_sequence",
    "full_context_sequence_padded",
    "cell",
    "base_editor",
    "energy_1",
    "energy_2",
    "energy_3",
    "energy_4",
    "energy_5",
    "energy_6",
    "energy_7",
    "energy_8",
    "energy_9",
    "energy_10",
    "energy_11",
    "energy_12",
    "energy_13",
    "energy_14",
    "energy_15",
    "energy_16",
    "energy_17",
    "energy_18",
    "energy_19",
    "energy_20",
    "energy_21",
    "energy_22",
    "energy_23",
    "energy_24",
    "free_energy",
    "melt_temperature_grna",
    "melt_temperature_target",
    "study_id"
]

# =====================================================
# Helper functions
# =====================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT,"r") as f:
            return json.load(f)
    return {}

def save_checkpoint(name, offset):
    checkpoint = load_checkpoint()
    checkpoint[name] = offset
    with open(CHECKPOINT,"w") as f:
        json.dump(
            checkpoint,
            f,
            indent=4
        )

def safe_request(endpoint, params):
    """
    API request with retry
    """
    for attempt in range(MAX_RETRY):
        try:
            result = api.get(
                endpoint,
                params
            )
            time.sleep(SLEEP_TIME)
            return result

        except Exception as e:
            error = str(e)
            if any(
                x in error
                for x in [
                    "429",
                    "502",
                    "503",
                    "504"
                ]
            ):
                wait = 60*(attempt+1)
                print(
                    f"[Server busy] "
                    f"{error}, "
                    f"wait {wait}s"
                )
                time.sleep(wait)
            else:
                raise e

    raise RuntimeError("Maximum retry exceeded")

# =====================================================
# Main downloader
# =====================================================
def download(endpoint, output_file, columns, name):
    print("\n")
    print("="*60)
    print(
        f"Downloading {name}"
    )
    print("="*60)

    checkpoint = load_checkpoint()
    offset = checkpoint.get(
        name,
        0
    )
    all_data=[]

    # resume existing file
    if os.path.exists(output_file):
        old = pd.read_csv(output_file)
        all_data.append(old)
        print(
            f"Loaded existing file "
            f"{old.shape}"
        )

    while True:
        print(
            f"{name}: "
            f"offset={offset}"
        )

        params = {
            "limit": LIMIT,
            "offset": offset,
            "base_editor": BASE_EDITOR
        }

        batch = safe_request(
            endpoint,
            params
        )

        if len(batch)==0:
            print("Finished")
            break

        df = pd.DataFrame(batch)

        print(
            "Received:",
            df.shape
        )

        all_data.append(df)
        offset += len(df)
        save_checkpoint(
            name,
            offset
        )

        # avoid huge memory
        temp = pd.concat(
            all_data,
            ignore_index=True
        )

        keep_cols = [
            c for c in columns
            if c in temp.columns
        ]

        temp[
            keep_cols
        ].to_csv(
            output_file,
            index=False
        )

        print(
            "Saved:",
            temp[keep_cols].shape
        )
        time.sleep(SLEEP_TIME)

    print(f"{name} completed")

# =====================================================
# Run
# =====================================================
if __name__=="__main__":
    download(
        endpoint="efficiency",
        output_file=EFF_OUTPUT,
        columns=EFF_COLUMNS,
        name="efficiency"
    )

    print("\nCooling down 120s...")
    time.sleep(120)

    # find bystander columns later
    download(
        endpoint="bystander",
        output_file=BYS_OUTPUT,
        columns=BYS_BASE_COLUMNS,
        name="bystander"
    )

    print("\nAll download finished.")