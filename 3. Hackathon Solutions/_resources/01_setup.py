# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./00_configuration

# COMMAND ----------

import os
import re
import mlflow
spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles", "10")

# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
  dbutils.fs.rm(cloud_storage_path, True)

spark.sql(f"""create database if not exists {db_name} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {db_name}""")

# COMMAND ----------

print(cloud_storage_path)
print(db_name)

# COMMAND ----------

path = cloud_storage_path

dirname = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
filename = "02_data_generator"
if (os.path.basename(dirname) != '_resources'):
  dirname = os.path.join(dirname,'_resources')
generate_data_notebook_path = os.path.join(dirname,filename)

def generate_data():
  dbutils.notebook.run(generate_data_notebook_path, 600, {"reset_all_data": reset_all, "db_name": db_name, "cloud_storage_path": cloud_storage_path})

if reset_all:
  generate_data()
else:
  try:
    dbutils.fs.ls(path)
  except: 
    generate_data()

# COMMAND ----------

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
mlflow.set_experiment('/Users/{}/parts_demand_forecasting'.format(current_user))

# COMMAND ----------


