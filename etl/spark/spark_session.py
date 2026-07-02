import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType
)

if len(sys.argv) < 2:
    print("Usage: client_transformer.py <bucket_name>")
    sys.exit(1)

BUCKET_NAME = sys.argv[1]
INPUT_PATH  = f"gs://{BUCKET_NAME}/raw/clients.csv"
OUTPUT_PATH = f"gs://{BUCKET_NAME}/delta/clients"

print(f"Input  : {INPUT_PATH}")
print(f"Output : {OUTPUT_PATH}")

spark = SparkSession.builder \
    .appName("ClientTransformer") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

schema = StructType([
    StructField("client_id",      StringType(),  False),
    StructField("name",           StringType(),  True),
    StructField("email",          StringType(),  True),
    StructField("phone",          StringType(),  True),
    StructField("city",           StringType(),  True),
    StructField("age",            IntegerType(), True),
    StructField("risk_profile",   StringType(),  True),
    StructField("aum",            FloatType(),   True),
    StructField("onboarded_date", StringType(),  True),
    StructField("advisor_id",     StringType(),  True),
    StructField("kyc_status",     StringType(),  True),
])

print("Reading clients.csv from GCS...")

df_raw = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("nullValue", "") \
    .schema(schema) \
    .load(INPUT_PATH)

raw_count = df_raw.count()
print(f"Raw rows: {raw_count:,}")

print("Cleaning data...")

df_clean = df_raw \
    .withColumn("name",
        F.initcap(F.trim(F.col("name")))) \
    .withColumn("email",
        F.lower(F.trim(F.col("email")))) \
    .withColumn("city",
        F.initcap(F.trim(F.col("city")))) \
    .withColumn("risk_profile",
        F.initcap(F.trim(F.col("risk_profile")))) \
    .withColumn("kyc_status",
        F.upper(F.trim(F.col("kyc_status")))) \
    .withColumn("onboarded_date",
        F.to_date(F.col("onboarded_date"), "yyyy-MM-dd")) \
    .fillna({
        "city": "Unknown",
        "risk_profile": "Moderate",
        "kyc_status": "PENDING",
        "advisor_id": "ADV000"
    }) \
    .filter(F.col("client_id").isNotNull()) \
    .filter(F.col("aum") > 0) \
    .filter(F.col("age").between(18, 100))

clean_count = df_clean.count()
print(f"Clean rows: {clean_count:,} (removed {raw_count - clean_count:,} invalid rows)")

print("Enriching data with derived columns...")

TIER1_CITIES = ["Mumbai", "Delhi", "Bangalore",
                "Chennai", "Hyderabad", "Kolkata"]

df_enriched = df_clean \
    .withColumn("age_group",
        F.when(F.col("age") < 35, "Young")
         .when(F.col("age") < 55, "Middle-Aged")
         .otherwise("Senior")) \
    .withColumn("aum_tier",
        F.when(F.col("aum") < 500_000, "Mass")
         .when(F.col("aum") < 5_000_000, "HNI")
         .otherwise("UHNI")) \
    .withColumn("years_since_onboard",
        F.round(
            F.datediff(F.current_date(),
                       F.col("onboarded_date")) / 365.25,
            1)) \
    .withColumn("city_tier",
        F.when(F.col("city").isin(TIER1_CITIES), "Tier1")
         .otherwise("Tier2")) \
    .withColumn("is_high_value",
        (F.col("aum") >= 1_000_000) &
        (F.col("kyc_status") == "VERIFIED")) \
    .withColumn("data_quality_score",
        F.when(F.col("email").isNotNull(), 1).otherwise(0) +
        F.when(F.col("phone").isNotNull(), 1).otherwise(0) +
        F.when(F.col("city") != "Unknown", 1).otherwise(0) +
        F.when(F.col("kyc_status") == "VERIFIED", 1).otherwise(0) +
        F.when(F.col("onboarded_date").isNotNull(), 1).otherwise(0))

print("\nSample enriched data:")
df_enriched.select(
    "client_id", "name", "city", "risk_profile",
    "age_group", "aum_tier", "is_high_value"
).show(5, truncate=False)

print(f"\nWriting to Delta Lake: {OUTPUT_PATH}")

df_enriched.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("risk_profile") \
    .save(OUTPUT_PATH)

print("Delta Lake write complete!")

print("\nVerification — reading back from Delta Lake:")

df_verify = spark.read.format("delta").load(OUTPUT_PATH)
total = df_verify.count()
print(f"Total rows in Delta table: {total:,}")

print("\nRisk profile distribution:")
df_verify.groupBy("risk_profile") \
         .count() \
         .orderBy("count", ascending=False) \
         .show()

print("\nAUM tier distribution:")
df_verify.groupBy("aum_tier") \
         .count() \
         .orderBy("count", ascending=False) \
         .show()

print(f"\nTotal AUM: Rs.{df_verify.agg(F.sum('aum')).collect()[0][0]/1e7:.1f} crore")

print("\nClientTransformer job complete!")
spark.stop()