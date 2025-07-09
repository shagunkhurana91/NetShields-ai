from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_date, when

spark = SparkSession.builder \
    .appName("GenAI Anomaly Detection - Feature Engineering") \
    .getOrCreate()
df=spark.read.parquet("C:/GenAI-Anomaly/outputs/cleaned_data.parquet")

df=df.withColumn("hour",hour("timestamp"))

df=df.withColumn("date",to_date("timestamp"))

df=df.withColumn("login_status",when(col("login_successful")==True,1).otherwise(0))

df.printSchema()
df.show(5)
df.write.mode("overwrite").parquet("C:/GenAI-Anomaly/outputs/engineered_data.parquet")
