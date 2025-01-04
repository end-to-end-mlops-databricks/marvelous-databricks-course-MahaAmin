from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("adb-3537333413571968").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)