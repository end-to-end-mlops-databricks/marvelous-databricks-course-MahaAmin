# Databricks notebook source
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install --force-reinstall /Volumes/fraud_credit_cards/packages/fraud_credit_cards/fraud_credit_cards-0.2.2-py3-none-any.whl

# COMMAND ----------

from pyspark.sql import SparkSession

from fraud_credit_cards.config import ProjectConfig
from fraud_credit_cards.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml("../project_config.yml")

# COMMAND ----------

config = ProjectConfig.from_yaml("../project_config.yml")


# COMMAND ----------

df = spark.read.csv(
    "/Volumes/fraud_credit_cards/data/credit_cards_2023/creditcard_2023.csv", header=True, inferSchema=True
).toPandas()

# COMMAND ----------

data_preprocessor = DataProcessor(pandas_df=df, config=config)

# COMMAND ----------

data_preprocessor.preprocess_data()

# COMMAND ----------

train_set, test_set = data_preprocessor.split_data()

# COMMAND ----------

data_preprocessor.save_to_catalog(train_set, test_set, spark=spark)

# COMMAND ----------
