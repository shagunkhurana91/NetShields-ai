from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, max as spark_max

spark = SparkSession.builder \
    .appName("GenAI Anomaly Detection - Behavior Aggregation") \
    .getOrCreate()

df = spark.read.parquet("C:/GenAI-Anomaly/outputs/engineered_data.parquet")

agg_df = df.groupBy("user_id", "date", "hour").agg(
    count("*").alias("total_logins"),
    spark_sum(1 - col("login_status")).alias("failed_logins"),
    spark_max("is_account_takeover").alias("is_account_takeover")
)

agg_df = agg_df.withColumn("flag_suspicious", (col("failed_logins") > 3).cast("int"))

agg_df.show(5)
agg_df.printSchema()
agg_df.write.mode("overwrite").parquet("C:/GenAI-Anomaly/outputs/user_login_flags.parquet")
print(" Saved aggregated flags to user_login_flags.parquet")
