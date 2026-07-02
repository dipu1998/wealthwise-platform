# =============================================================
# FILE: ml/feature_export.py
#
# READS DELTA LAKE PARQUET FILES DIRECTLY WITH PANDAS
# Bypasses BigQuery entirely — no partition column issue
# =============================================================

import os
import pandas as pd
from google.cloud import storage

PROJECT_ID  = "project-0ef73ac4-7a69-455a-987"
BUCKET_NAME = f"{PROJECT_ID}-wealthwise"

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

def read_delta_as_df(prefix):
    """
    Lists all .parquet files under a GCS prefix,
    downloads each one, reads with pandas, concatenates.
    The partition folder name (risk_profile=Conservative)
    is parsed to recover the partition column value.
    """
    blobs = list(bucket.list_blobs(prefix=prefix))
    parquet_blobs = [b for b in blobs if b.name.endswith(".parquet")]
    print(f"  Found {len(parquet_blobs)} parquet files under gs://{BUCKET_NAME}/{prefix}")

    frames = []
    for blob in parquet_blobs:
        # Extract partition value from folder name
        # e.g. "delta/clients/risk_profile=Conservative/part-00000.parquet"
        #  → risk_profile = "Conservative"
        parts = blob.name.split("/")
        partition_value = None
        for part in parts:
            if "=" in part:
                partition_value = part.split("=")[1]

        # Download parquet bytes into pandas
        data = blob.download_as_bytes()
        import io
        df = pd.read_parquet(io.BytesIO(data))

        # Add the partition column back
        if partition_value:
            df["risk_profile"] = partition_value

        frames.append(df)

    return pd.concat(frames, ignore_index=True)

# ── READ CLIENTS ──────────────────────────────────────────────
print("Reading clients Delta table...")
df_clients = read_delta_as_df("delta/clients/")
print(f"  Clients rows: {len(df_clients):,}")
print(f"  Columns: {list(df_clients.columns)}")

# ── READ PORTFOLIOS ───────────────────────────────────────────
print("\nReading portfolios Delta table...")
df_portfolios = read_delta_as_df("delta/portfolios/")
print(f"  Portfolio rows: {len(df_portfolios):,}")
print(f"  Columns: {list(df_portfolios.columns)}")

# ── JOIN ──────────────────────────────────────────────────────
print("\nJoining clients + portfolios...")

df_clients_slim = df_clients[["client_id", "aum", "age_group"]].copy()
df_portfolios_slim = df_portfolios[[
    "client_id", "equity_pct", "debt_pct", "gold_pct",
    "total_num_funds", "years_invested", "risk_profile"
]].copy()

df = df_portfolios_slim.merge(df_clients_slim, on="client_id", how="inner")

# ── FEATURE ENGINEERING ───────────────────────────────────────
df["aum_lakh"] = (df["aum"] / 100_000).round(2)

df["age_group_encoded"] = df["age_group"].map({
    "Young": 0,
    "Middle-Aged": 1,
    "Senior": 2
}).fillna(1)

# Keep only clean rows
df = df[df["risk_profile"].isin(["Conservative", "Moderate", "Aggressive"])]
df = df.dropna(subset=["equity_pct", "debt_pct", "gold_pct"])

# Final feature columns
df_final = df[[
    "client_id",
    "equity_pct",
    "debt_pct",
    "gold_pct",
    "total_num_funds",
    "years_invested",
    "aum_lakh",
    "age_group_encoded",
    "risk_profile"
]].rename(columns={"risk_profile": "target"})

print(f"\nFinal training dataset: {len(df_final):,} rows")
print(f"\nTarget distribution:")
print(df_final["target"].value_counts())
print(f"\nSample:")
print(df_final.head(5).to_string())

# ── SAVE ──────────────────────────────────────────────────────
os.makedirs("ml/data", exist_ok=True)
output_path = "ml/data/training_features.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print("Feature export complete!")