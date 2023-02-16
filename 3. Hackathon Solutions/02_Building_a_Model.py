# Databricks notebook source
# MAGIC %md
# MAGIC *Prerequisite: Complete 01_Hyperparameter_Tuning before running this notebook.*
# MAGIC 
# MAGIC In this notebook we leverage the exploration in notebook 01 to tune build a scalable model in a distributed way 
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Pandas UDFs (user-defined functions) can take your single-node data science code, and distribute it across different keys (e.g. SKU)  
# MAGIC - Hyperopt can also perform hyperparameter tuning from within a Pandas UDF  

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

dbName = 'demand_planning_josh_melton'
print(dbName)

# COMMAND ----------

import os
import datetime as dt
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)

from statsmodels.tsa.api import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_percentage_error

import pyspark.sql.functions as f
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Read in data

# COMMAND ----------

demand_df = spark.read.table(f"{dbName}.part_level_demand")
demand_df = demand_df.cache() # just for this example notebook

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Train thousands of models at scale, any time
# MAGIC *while still using your preferred libraries and approaches*

# COMMAND ----------

FORECAST_HORIZON = 40 # TODO: choose a number for the forecast horizon (in weeks). We recommend somewhere between 26-52

# COMMAND ----------

#JOSH TODO: format this example
# def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
#     """
#     Applies model to data for a particular device, represented as a pandas DataFrame
#     """
#     model_path = df_pandas["model_path"].iloc[0]

#     input_columns = ["feature_1", "feature_2", "feature_3"]
#     X = df_pandas[input_columns]

#     model = mlflow.sklearn.load_model(model_path)
#     prediction = model.predict(X)

#     return_df = pd.DataFrame({
#         "record_id": df_pandas["record_id"],
#         "prediction": prediction
#     })
#     return return_df

# COMMAND ----------

# DBTITLE 0,Modularize single-node logic from before into Python functions
def add_exo_variables(pdf: pd.DataFrame) -> pd.DataFrame: # TODO: specify what will be the output
  
  midnight = dt.datetime.min.time()
  timestamp = pdf["Date"].apply(lambda x: dt.datetime.combine(x, midnight))
  calendar_week = timestamp.dt.isocalendar().week
  
  # define flexible, custom logic for exogenous variables
  covid_breakpoint = dt.datetime(year=2020, month=3, day=1)
  enriched_df = (
    pdf
    .assign(covid = (timestamp >= covid_breakpoint).astype(float))
    .assign(christmas = ((calendar_week >= 51) & (calendar_week <= 52)).astype(float))
    .assign(new_year = ((calendar_week >= 1) & (calendar_week <= 4)).astype(float))
  )
  return enriched_df[["Date", "Product", "SKU", "Demand", "covid", "christmas", "new_year"]]

# COMMAND ----------

def split_train_score_data(data, forecast_horizon=FORECAST_HORIZON):
  """
  - assumes data is sorted by date/time already
  - forecast_horizon in weeks
  """
  is_history = [True] * (len(data) - forecast_horizon) + [False] * forecast_horizon
  train = data.iloc[is_history]
  score = data.iloc[~np.array(is_history)]
  return train, score

# COMMAND ----------

# Expected output for our enriched dataset
enriched_schema = StructType(
  [
    StructField('Date', DateType()),
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('Demand', FloatType()),
    StructField('covid', FloatType()),
    StructField('christmas', FloatType()),
    StructField('new_year', FloatType()),
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a look at our core dataset, with exogenous variables added. Data for all SKUs is logically unified within a Spark DataFrame, allowing large-scale distributed processing rather than single node procecssing with pandas.

# COMMAND ----------

enriched_df = (
  demand_df
    .groupBy("Product")
    .applyInPandas(add_exo_variables, enriched_schema) # TODO: use applyInPandas to run our add_exo_variables function in a distributed way
)
display(enriched_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### What we're doing: High-Level Overview
# MAGIC 
# MAGIC Benefits:
# MAGIC - Pure Python & Pandas: easy to develop, test
# MAGIC - Continue using your favorite libraries
# MAGIC - Simply assume you're working with a Pandas DataFrame for a single SKU
# MAGIC 
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/pandas-udf-workflow.png?raw=true" width=40%>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Build, tune and score a model per each SKU with Pandas UDFs

# COMMAND ----------

# JOSH TODO: add formatted example

# COMMAND ----------

def build_tune_and_score_model(sku_pdf: pd.DataFrame) -> pd.DataFrame: # TODO: what is expected the input and output format?
    """
    This function trains, tunes and scores a model for each SKU and can be distributed as a Pandas UDF
    """
    # Always ensure proper ordering and indexing by Date
    sku_pdf.sort_values("Date", inplace=True)
    complete_ts = sku_pdf.set_index("Date").asfreq(freq="W-MON")
    
    print(complete_ts)
    

    # Since we'll group the large Spark DataFrame by (Product, SKU)
    PRODUCT = sku_pdf["Product"].iloc[0]
    SKU = sku_pdf["SKU"].iloc[0]
    train_data, validation_data = split_train_score_data(complete_ts)
    exo_fields = ["covid", "christmas", "new_year"]


    # Evaluate model on the traing data set
    def evaluate_model(hyperopt_params):

          # SARIMAX requires a tuple of Python integers
          order_hparams = tuple([int(hyperopt_params[k]) for k in ("p", "d", "q")])

          # Training
          model = SARIMAX(
            train_data["Demand"], 
            exog=train_data[exo_fields], 
            order=order_hparams, 
            seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
            initialization_method="estimated",
            enforce_stationarity = False,
            enforce_invertibility = False
          )
          fitted_model = model.fit(disp=False, method='nm') # TODO: fit the model defined above to our data

          # Validation
          fcast = fitted_model.predict(  # TODO: define how the model should make predictions
            start=validation_data.index.min(), 
            end=validation_data.index.max(), 
            exog=validation_data[exo_fields]
          )

          return {'status': hyperopt.STATUS_OK, 'loss': np.power(validation_data.Demand.to_numpy() - fcast.to_numpy(), 2).mean()}

    search_space = {
        'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)), # TODO: define the distribution of the hyperopt search space
        'd': scope.int(hyperopt.hp.quniform('d', 0, 2, 1)), # TODO: define the distribution of the hyperopt search space
        'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1))  # TODO: define the distribution of the hyperopt search space
    }

    rstate = np.random.default_rng(123) # just for reproducibility of this notebook

    best_hparams = fmin(   # TODO: provide the hyperopt function that defines what we're minimizing
      fn=evaluate_model,   # TODO: provide the function that we'll use to evaluate the model (defined in line 20)
      space=search_space,  # TODO: provide the search space
      algo=tpe.suggest, 
      max_evals=10
    )

    # Training
    model_final = SARIMAX(
      train_data["Demand"], 
      exog=train_data[exo_fields], 
      order=tuple(best_hparams.values()), 
      seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
      initialization_method="estimated",
      enforce_stationarity = False,
      enforce_invertibility = False
    )
    fitted_model_final = model_final.fit(disp=False, method='nm')

    # Validation
    fcast = fitted_model_final.predict(
      start=complete_ts.index.min(), 
      end=complete_ts.index.max(), 
      exog=validation_data[exo_fields]
    )

    forecast_series = complete_ts[['Product', 'SKU' , 'Demand']].assign(Date = complete_ts.index.values).assign(Demand_Fitted = fcast)
      
    forecast_series = forecast_series[['Product', 'SKU' , 'Date', 'Demand', 'Demand_Fitted']]
    
    return forecast_series

# COMMAND ----------

tuning_schema = StructType(
  [
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('Date', DateType()),
    StructField('Demand', FloatType()),
    StructField('Demand_Fitted', FloatType())
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Run distributed processing: `groupBy("SKU")` + `applyInPandas(...)`

# COMMAND ----------

# JOSH TODO: add formatted example

# COMMAND ----------

# To maximize parallelism, we can allocate each ("Product", SKU") group its own Spark task.
# First, we'll disable Adaptive Query Execution (AQE) just for this step
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")

# next, partition our dataframe by Product+SKU combinations
n_tasks = enriched_df.select("Product", "SKU").distinct().count()

forecast_df = (
  enriched_df
  .repartition(n_tasks, "Product", "SKU")
  .groupBy("Product", "SKU") # TODO: group by the columns you're partitioning by
  .applyInPandas(build_tune_and_score_model, schema=tuning_schema) # TODO: apply the model function that was defined in pandas
)

display(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

# Write the data 
forecast_df.write \
.mode("overwrite") \
.format("delta") \
.saveAsTable(f'{dbName}.part_level_demand_with_forecasts')

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dbName}.part_level_demand_with_forecasts"))

# COMMAND ----------


