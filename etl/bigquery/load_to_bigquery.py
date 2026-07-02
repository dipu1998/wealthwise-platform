# =============================================================
# FILE: etl/bigquery/load_to_bigquery.py
#
# WHAT IT DOES:
#   Reads the two Delta Lake tables from GCS
#   Loads them into BigQuery as external tables
#
# WHY EXTERNAL TABLES?
#   Two options to get Delta data into BigQuery:
#
#   Option A — Copy data INTO BigQuery (managed table):
#     BigQuery stores its own copy. Fast queries.
#     But data is duplicated. If Delta updates, BigQuery is stale.
#
#   Option B — External table (what we use):
#     BigQuery points to the GCS files directly.
#     No data copy. Always fresh.
#     Slightly slower queries but zero duplication.
#
#   In production: use external table for raw/delta data,
#   use managed tables for aggregated/reporting data.
#
# COMPANY USE CASE:
#   Data analysts query BigQuery with SQL — they don't know Spark
#   Dashboards (Looker/Data Studio) connect to BigQuery
#   ML feature pipelines export from BigQuery to training jobs
# =============================================================

from google.cloud import bigquery

PROJECT_ID  = "project-0ef73ac4-7a69-455a-987"
DATASET_ID  = "wealthwise_db"
BUCKET_NAME = f"{PROJECT_ID}-wealthwise"

client = bigquery.Client(project=PROJECT_ID)

# ── STEP 1: Create Dataset ────────────────────────────────────
# A dataset in BigQuery = a folder/schema that holds tables
# Like a database in MySQL

print(f"Creating dataset {DATASET_ID}...")
dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
dataset_ref.location = "US"

try:
    client.create_dataset(dataset_ref)
    print(f"  Dataset {DATASET_ID} created")
except Exception as e:
    if "Already Exists" in str(e):
        print(f"  Dataset {DATASET_ID} already exists — skipping")
    else:
        raise e

# ── STEP 2: Create External Tables ───────────────────────────
# External table = BigQuery reads files directly from GCS
# format="PARQUET" works for both Parquet and Delta Lake files
# (Delta Lake stores data as Parquet files + transaction log)

tables = [
    {
        "table_id": "clients",
        "gcs_uri":  f"gs://{BUCKET_NAME}/delta/clients/*.parquet",
        "description": "Client master data — cleaned and enriched by Spark"
    },
    {
        "table_id": "portfolios",
        "gcs_uri":  f"gs://{BUCKET_NAME}/delta/portfolios/*.parquet",
        "description": "Portfolio summaries per client — aggregated by Spark"
    },
]
for t in tables:
    full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{t['table_id']}"
    print(f"\nCreating external table {full_table_id}...")

    # ExternalConfig tells BigQuery:
    # "Don't store the data yourself — read it from this GCS path"
    external_config = bigquery.ExternalConfig("PARQUET")
    external_config.source_uris = [t["gcs_uri"]]
    external_config.autodetect = True   # BigQuery infers schema from files

    table = bigquery.Table(full_table_id)
    table.external_data_configuration = external_config
    table.description = t["description"]

    try:
        client.delete_table(full_table_id)
        print(f"  Deleted old version")
    except Exception:
        pass  # table didn't exist — fine

    client.create_table(table)
    print(f"  Created: {full_table_id}")

# ── STEP 3: Verify with a Query ───────────────────────────────
print("\n\nVerification queries:")

queries = [
    {
        "label": "Total clients count",
        "sql": f"""
            SELECT COUNT(*) as total_clients,
                   ROUND(AVG(aum)/100000, 1) as avg_aum_lakh,
                   ROUND(SUM(aum)/10000000, 1) as total_aum_crore
            FROM `{PROJECT_ID}.{DATASET_ID}.clients`
        """
    },
    {
        "label": "AUM tier distribution",
        "sql": f"""
            SELECT aum_tier,
                   COUNT(*) as total_clients,
                   ROUND(SUM(aum)/10000000, 1) as total_aum_crore
            FROM `{PROJECT_ID}.{DATASET_ID}.clients`
            GROUP BY aum_tier
            ORDER BY total_aum_crore DESC
        """
    },
    {
        "label": "Portfolio allocation averages",
        "sql": f"""
            SELECT
                ROUND(AVG(equity_pct), 1) as avg_equity_pct,
                ROUND(AVG(debt_pct), 1) as avg_debt_pct,
                ROUND(AVG(gold_pct), 1) as avg_gold_pct,
                ROUND(AVG(total_num_funds), 1) as avg_funds,
                COUNT(*) as total_clients
            FROM `{PROJECT_ID}.{DATASET_ID}.portfolios`
        """
    },
]

for q in queries:
    print(f"\n--- {q['label']} ---")
    rows = client.query(q["sql"]).result()
    for row in rows:
        print(dict(row))

print("\n\nBigQuery load complete!")
print(f"View tables at: https://console.cloud.google.com/bigquery?project={PROJECT_ID}")