"""
Process Full Edge-IIoT Dataset for Federated Learning.

Reads all 24 CSV files (10 normal sensor types + 14 attack types),
combines them, cleans features, and partitions into 10 FL clients
based on sensor type (natural non-IID distribution).

Each sensor type becomes a natural FL client, with attack traffic
distributed proportionally across all clients.

Output: data/edge_full/client_*.parquet + metadata.json
"""

import os
import sys
import json
import gc
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATASET_ROOT = os.environ.get("EDGE_IIOT_RAW_DIR", "data/raw/edge")
NORMAL_DIR = os.path.join(DATASET_ROOT, "Normal traffic")
ATTACK_DIR = os.path.join(DATASET_ROOT, "Attack traffic")
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "edge_full"
)

# Sensor types (these become our 10 FL clients)
SENSOR_TYPES = [
    "Distance", "Flame_Sensor", "Heart_Rate", "IR_Receiver",
    "Modbus", "phValue", "Soil_Moisture", "Sound_Sensor",
    "Temperature_and_Humidity", "Water_Level"
]

# Columns to drop (identifiers, timestamps, non-numeric strings)
DROP_COLUMNS = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.dst.proto_ipv4", "arp.src.proto_ipv4",
    "dns.qry.name", "http.file_data", "http.request.uri.query",
    "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "mqtt.protoname", "mqtt.msg",
    "mqtt.msg_decoded_as", "mqtt.topic",
    "Attack_type",  # Keep only binary label
]

# Column that stays as label
LABEL_COLUMN = "Attack_label"


def log(msg):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_csv_safe(filepath, label_override=None):
    """Load a CSV with error handling, converting to numeric."""
    log(f"  Loading: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        log(f"    Rows: {len(df):,}")

        # Standardize Attack_label
        if LABEL_COLUMN in df.columns:
            # Normal traffic has Attack_label=0 or "Normal", Attack has 1
            df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce').fillna(0).astype(int)
            # Ensure binary: anything > 0 is attack
            df[LABEL_COLUMN] = (df[LABEL_COLUMN] > 0).astype(int)

        if label_override is not None and LABEL_COLUMN in df.columns:
            df[LABEL_COLUMN] = label_override

        return df
    except Exception as e:
        log(f"    ERROR: {e}")
        return None


def clean_features(df):
    """Drop non-numeric columns and clean the dataframe."""
    # Drop specified columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Separate label
    label_col = None
    if LABEL_COLUMN in df.columns:
        label_col = df[LABEL_COLUMN].copy()
        df = df.drop(columns=[LABEL_COLUMN])

    # Convert all remaining columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN/Inf with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Re-attach label
    if label_col is not None:
        df[LABEL_COLUMN] = label_col.values

    return df


def process():
    """Main processing pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 60)
    log("PHASE 1: Loading Normal Traffic (10 sensor types)")
    log("=" * 60)

    sensor_dfs = {}
    for sensor in SENSOR_TYPES:
        sensor_dir = os.path.join(NORMAL_DIR, sensor)
        csv_file = os.path.join(sensor_dir, f"{sensor}.csv")

        if not os.path.exists(csv_file):
            log(f"  WARNING: {csv_file} not found, skipping")
            continue

        df = load_csv_safe(csv_file, label_override=0)  # Normal = 0
        if df is not None:
            df = clean_features(df)
            sensor_dfs[sensor] = df
            log(f"    Clean shape: {df.shape}")

    log(f"\nLoaded {len(sensor_dfs)} sensor types")

    # Get the common feature set across all normal files
    all_feature_cols = None
    for sensor, df in sensor_dfs.items():
        cols = set(df.columns) - {LABEL_COLUMN}
        if all_feature_cols is None:
            all_feature_cols = cols
        else:
            all_feature_cols = all_feature_cols & cols

    log(f"Common features across all sensors: {len(all_feature_cols)}")
    feature_cols = sorted(list(all_feature_cols))

    log("\n" + "=" * 60)
    log("PHASE 2: Loading Attack Traffic (14 attack types)")
    log("=" * 60)

    attack_dfs = []
    attack_files = [f for f in os.listdir(ATTACK_DIR) if f.endswith('.csv')]

    for attack_file in sorted(attack_files):
        filepath = os.path.join(ATTACK_DIR, attack_file)
        df = load_csv_safe(filepath, label_override=1)  # Attack = 1
        if df is not None:
            df = clean_features(df)
            # Keep only common features + label
            keep_cols = [c for c in feature_cols if c in df.columns] + [LABEL_COLUMN]
            df = df[keep_cols]
            attack_dfs.append(df)
            log(f"    Clean shape: {df.shape}")

    if attack_dfs:
        all_attacks = pd.concat(attack_dfs, ignore_index=True)
        log(f"\nTotal attack rows: {len(all_attacks):,}")
        del attack_dfs
        gc.collect()
    else:
        log("WARNING: No attack traffic loaded!")
        all_attacks = pd.DataFrame()

    log("\n" + "=" * 60)
    log("PHASE 3: Partitioning into FL Clients")
    log("=" * 60)

    # Strategy: Each sensor gets its normal traffic + proportional share of attacks
    # This creates natural non-IID distribution (each sensor has different traffic patterns)
    # but shares the same attack distribution

    total_normal_rows = sum(len(df) for df in sensor_dfs.values())
    num_clients = len(sensor_dfs)

    metadata = {}
    client_idx = 0

    for sensor in SENSOR_TYPES:
        if sensor not in sensor_dfs:
            continue

        normal_df = sensor_dfs[sensor]
        # Keep only common features
        keep_cols = [c for c in feature_cols if c in normal_df.columns] + [LABEL_COLUMN]
        normal_df = normal_df[keep_cols]

        # Proportional attack allocation based on normal traffic size
        proportion = len(normal_df) / total_normal_rows
        num_attack_rows = int(len(all_attacks) * proportion)

        # Sample attack rows for this client (without replacement across clients)
        if len(all_attacks) > 0 and num_attack_rows > 0:
            attack_sample = all_attacks.sample(
                n=min(num_attack_rows, len(all_attacks)),
                random_state=42 + client_idx
            )
        else:
            attack_sample = pd.DataFrame()

        # Combine normal + attack for this client
        if len(attack_sample) > 0:
            client_df = pd.concat([normal_df, attack_sample], ignore_index=True)
        else:
            client_df = normal_df.copy()

        # Shuffle
        client_df = client_df.sample(frac=1, random_state=42 + client_idx).reset_index(drop=True)

        # Save
        client_name = f"client_{client_idx}"
        outpath = os.path.join(OUTPUT_DIR, f"{client_name}.parquet")
        client_df.to_parquet(outpath, index=False)

        benign = int((client_df[LABEL_COLUMN] == 0).sum())
        attack = int((client_df[LABEL_COLUMN] == 1).sum())

        metadata[str(client_idx)] = {
            "file": f"{client_name}.parquet",
            "sensor": sensor,
            "rows": len(client_df),
            "label_dist": {"0": benign, "1": attack},
            "benign_pct": round(100 * benign / len(client_df), 1),
        }

        log(f"  Client {client_idx} ({sensor}): {len(client_df):,} rows "
            f"(Benign: {benign:,}, Attack: {attack:,})")

        del client_df
        gc.collect()
        client_idx += 1

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "=" * 60)
    log("PHASE 4: Summary")
    log("=" * 60)

    total_rows = sum(m["rows"] for m in metadata.values())
    total_benign = sum(m["label_dist"]["0"] for m in metadata.values())
    total_attack = sum(m["label_dist"]["1"] for m in metadata.values())
    num_features = len(feature_cols)

    log(f"Clients: {len(metadata)}")
    log(f"Total rows: {total_rows:,}")
    log(f"Benign: {total_benign:,} ({100*total_benign/total_rows:.1f}%)")
    log(f"Attack: {total_attack:,} ({100*total_attack/total_rows:.1f}%)")
    log(f"Features: {num_features}")
    log(f"Output: {OUTPUT_DIR}")

    # Save feature list for reference
    with open(os.path.join(OUTPUT_DIR, "features.json"), "w") as f:
        json.dump({"features": feature_cols, "num_features": num_features}, f, indent=2)

    log("\nDone!")


if __name__ == "__main__":
    process()
