from pyspark.sql import SparkSession

print("🚀 Script started")

spark = SparkSession.builder \
    .appName("GenAI Anomaly Detection") \
    .getOrCreate()

df = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .load("C:/GenAI-Anomaly/data/rba-dataset.csv")

print("✅ Data Loaded")

df.printSchema()

print("\n🧾 Columns:")
print(df.columns)

print("\n🔍 Sample Rows:")
df.show(5)
print("\n📊 Total Rows:", df.count())
