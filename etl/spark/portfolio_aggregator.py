# =============================================================
# FILE: etl/spark/portfolio_aggregator.py
#
# WHAT IT DOES:
#   Reads raw portfolios.csv from GCS
#   Calculates per-client portfolio summary:
#     - Total AUM across all funds
#     - % in equity, debt, gold
#     - Number of funds held
#     - Gain/loss percentage
#   Writes result as Delta Lake partitioned by risk_profile
#
# WHY THIS FILE EXISTS:
#   One client can have 3-5 different fund investments
#   ML model needs ONE row per client with aggregated features
#   This job collapses multiple portfolio rows → one summary row
#
# COMPANY USE CASE:
#   Portfolio dashboard shows client's overall allocation
#   Risk model uses equity_pct to classify risk appetite
#   Advisor sees at a glance: "Client is 80% in equity = aggressive"
# =============================================================

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, IntegerType
)

if len(sys.argv) < 2:
    print("Usage: portfolio_aggregator.py <bucket_name>")
    sys.exit(1)

BUCKET_NAME      = sys.argv[1]
INPUT_PATH       = f"gs://{BUCKET_NAME}/raw/portfolios.csv"
CLIENTS_PATH     = f"gs://{BUCKET_NAME}/delta/clients"
OUTPUT_PATH      = f"gs://{BUCKET_NAME}/delta/portfolios"

print(f"Input  : {INPUT_PATH}")
print(f"Output : {OUTPUT_PATH}")

# ── SPARK SESSION ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("PortfolioAggregator") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# ── SCHEMA ────────────────────────────────────────────────────
# portfolios.csv has one row per fund investment
# One client can appear multiple times (one row per fund)

portfolio_schema = StructType([
    StructField("portfolio_id",  StringType(),  False),
    StructField("client_id",     StringType(),  False),
    StructField("fund_id",       StringType(),  True),
    StructField("fund_type",     StringType(),  True),  # equity/debt/gold
    StructField("invested_amt",  FloatType(),   True),
    StructField("current_value", FloatType(),   True),
    StructField("num_units",     FloatType(),   True),
    StructField("purchase_date", StringType(),  True),
])

# ── READ RAW PORTFOLIOS ───────────────────────────────────────
print("Reading portfolios.csv from GCS...")

df_raw = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("nullValue", "") \
    .schema(portfolio_schema) \
    .load(INPUT_PATH)

raw_count = df_raw.count()
print(f"Raw portfolio rows: {raw_count:,}")

# ── CLEAN ─────────────────────────────────────────────────────
df_clean = df_raw \
    .withColumn("fund_type",
        F.lower(F.trim(F.col("fund_type")))) \
    .withColumn("purchase_date",
        F.to_date(F.col("purchase_date"), "yyyy-MM-dd")) \
    .filter(F.col("client_id").isNotNull()) \
    .filter(F.col("invested_amt") > 0) \
    .filter(F.col("current_value") > 0) \
    .filter(F.col("fund_type").isin("equity", "debt", "gold"))

clean_count = df_clean.count()
print(f"Clean portfolio rows: {clean_count:,}")

# ── AGGREGATE PER CLIENT ──────────────────────────────────────
# WHY AGGREGATE?
# Raw data: Client CLT001 has 4 rows (4 different funds)
# We need: Client CLT001 has 1 row with total AUM, % breakdown
#
# groupBy("client_id") + pivot("fund_type") =
#   Takes multiple rows per client and rotates fund_type into columns
#
# BEFORE pivot:
#   CLT001 | equity | 500000
#   CLT001 | debt   | 300000
#   CLT001 | gold   | 200000
#
# AFTER pivot:
#   CLT001 | equity_amt=500000 | debt_amt=300000 | gold_amt=200000

print("Aggregating portfolio data per client...")

# Step 1: Total invested and current value per client per fund type
df_by_type = df_clean.groupBy("client_id", "fund_type").agg(
    F.sum("invested_amt").alias("invested"),
    F.sum("current_value").alias("current"),
    F.count("portfolio_id").alias("num_funds_in_type")
)

# Step 2: Pivot fund_type into columns
df_pivot = df_by_type.groupBy("client_id").pivot(
    "fund_type", ["equity", "debt", "gold"]
).agg(
    F.first("current")  # current value for that fund type
)

# Rename pivot columns to be clear
df_pivot = df_pivot \
    .withColumnRenamed("equity", "equity_value") \
    .withColumnRenamed("debt",   "debt_value") \
    .withColumnRenamed("gold",   "gold_value") \
    .fillna(0.0, subset=["equity_value", "debt_value", "gold_value"])

# Step 3: Total level aggregations
df_totals = df_clean.groupBy("client_id").agg(
    F.sum("invested_amt").alias("total_invested"),
    F.sum("current_value").alias("total_current_value"),
    F.count("portfolio_id").alias("total_num_funds"),
    F.min("purchase_date").alias("first_investment_date"),
    F.max("purchase_date").alias("latest_investment_date"),
)

# Step 4: Join pivot + totals
df_combined = df_totals.join(df_pivot, on="client_id", how="left")

# Step 5: Calculate percentages and gain/loss
# WHY PERCENTAGES?
# ML model works better with % than absolute amounts
# A client with 80% equity is aggressive whether they have 1L or 100Cr
# Percentages normalise the data → better model accuracy

df_features = df_combined \
    .withColumn("total_current_value_safe",
        F.when(F.col("total_current_value") == 0, 1.0)
         .otherwise(F.col("total_current_value"))) \
    .withColumn("equity_pct",
        F.round(F.col("equity_value") / F.col("total_current_value_safe") * 100, 2)) \
    .withColumn("debt_pct",
        F.round(F.col("debt_value") / F.col("total_current_value_safe") * 100, 2)) \
    .withColumn("gold_pct",
        F.round(F.col("gold_value") / F.col("total_current_value_safe") * 100, 2)) \
    .withColumn("gain_loss_pct",
        F.round(
            (F.col("total_current_value") - F.col("total_invested"))
            / F.col("total_invested") * 100,
            2)) \
    .withColumn("years_invested",
        F.round(
            F.datediff(F.current_date(), F.col("first_investment_date")) / 365.25,
            1)) \
    .drop("total_current_value_safe")

# ── JOIN WITH CLIENTS DELTA TABLE ─────────────────────────────
# WHY JOIN?
# Portfolio table has client_id but not risk_profile or aum_tier
# We need risk_profile to partition the output Delta table
# We read it from the clients Delta table we created in Step 1

print("Joining with clients Delta table for risk_profile...")

df_clients = spark.read.format("delta").load(CLIENTS_PATH) \
    .select("client_id", "risk_profile", "aum_tier", "age_group", "city_tier")

df_final = df_features.join(df_clients, on="client_id", how="left") \
    .fillna("Moderate", subset=["risk_profile"])

final_count = df_final.count()
print(f"Final portfolio summary rows: {final_count:,}")

# ── SHOW SAMPLE ───────────────────────────────────────────────
print("\nSample portfolio summary:")
df_final.select(
    "client_id", "total_invested", "total_current_value",
    "equity_pct", "debt_pct", "gold_pct",
    "gain_loss_pct", "total_num_funds", "risk_profile"
).show(5, truncate=False)

# ── WRITE TO DELTA LAKE ───────────────────────────────────────
print(f"\nWriting to Delta Lake: {OUTPUT_PATH}")

df_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("risk_profile") \
    .save(OUTPUT_PATH)

print("Delta Lake write complete!")

# ── VERIFY ────────────────────────────────────────────────────
print("\nVerification:")
df_verify = spark.read.format("delta").load(OUTPUT_PATH)
print(f"Total clients with portfolio summary: {df_verify.count():,}")

print("\nAverage allocation by risk profile:")
df_verify.groupBy("risk_profile").agg(
    F.round(F.avg("equity_pct"), 1).alias("avg_equity_pct"),
    F.round(F.avg("debt_pct"), 1).alias("avg_debt_pct"),
    F.round(F.avg("gold_pct"), 1).alias("avg_gold_pct"),
    F.round(F.avg("gain_loss_pct"), 1).alias("avg_gain_loss_pct"),
).orderBy("risk_profile").show()

print("\nPortfolioAggregator job complete!")
spark.stop()