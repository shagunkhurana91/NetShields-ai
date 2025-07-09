from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

spark = SparkSession.builder \
    .appName("GenAI Anomaly Detection - Clean Data") \
    .getOrCreate()

df = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .load("C:/GenAI-Anomaly/data/rba-dataset.csv")

df_clean = df.select(
    col("Login Timestamp").alias("timestamp"),
    col("User ID").alias("user_id"),
    col("IP Address").alias("ip_address"),
    col("Country").alias("country"),
    col("Login Successful").alias("login_successful"),
    col("Is Attack IP").alias("is_attack_ip"),
    col("Is Account Takeover").alias("is_account_takeover")
)
df_clean = df_clean.withColumn("timestamp", to_timestamp("timestamp"))

df_clean.printSchema()
df_clean.show(5)
print(f" Cleaned Row Count: {df_clean.count()}")
df_clean.write.mode("overwrite").parquet("C:/GenAI-Anomaly/outputs/cleaned_data.parquet")
print("Cleaned data saved.")

