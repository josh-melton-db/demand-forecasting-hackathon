# Databricks notebook source
# MAGIC %md
# MAGIC *Prerequisite: Complete 01_Hyperparameter_Tuning before running this notebook.*
# MAGIC 
# MAGIC Exercises: for the rest of this notebook, find the ```#TODO```s and fill in the ```...``` with your answers </br> 
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Hyperopt for hyperparameter tuning 
# MAGIC - Take your single-node data science code from the last notebook, and distribute it across different keys (e.g. SKU) with Pandas UDFs

# COMMAND ----------

# MAGIC %run ./_resources/00_configuration

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

import pyspark.sql.functions as f
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Read in data

# COMMAND ----------

demand_df = spark.read.table(f"{db_name}.part_level_demand")
demand_df = demand_df.cache() # just for this example notebook

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC What we're doing, high-level overview:
# MAGIC 
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/pandas-udf-workflow.png?raw=true" width=40%>
# MAGIC 
# MAGIC Benefits:
# MAGIC - Pure Python & Pandas: easy to develop, test
# MAGIC - Continue using your favorite libraries
# MAGIC - Simply assume you're working with a Pandas DataFrame for a single SKU

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Transform data at scale, while still using your preferred libraries and approaches

# COMMAND ----------

FORECAST_HORIZON = ... # TODO: choose a number for the forecast horizon (in weeks). We recommend somewhere between 26-52

# COMMAND ----------

# MAGIC %md
# MAGIC Example input/output type definition for a function to be run via <a href="https://spark.apache.org/docs/3.2.1/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html">applyInPandas</a>:
# MAGIC ```
# MAGIC def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
# MAGIC     ... transformations to df_pandas go here ...
# MAGIC     return return_df
# MAGIC ```

# COMMAND ----------

# DBTITLE 0,Modularize single-node logic from before into Python functions
# This function will be used to transform our data in parallel
def add_exo_variables(pdf: ...) -> ...: # TODO: specify what will be the input and output types
  
  midnight = dt.datetime.min.time()
  timestamp = pdf["Date"].apply(lambda x: dt.datetime.combine(x, midnight))
  calendar_week = timestamp.dt.isocalendar().week
  
  # define our exogenous variables
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

# Expected output for our enriched dataset, to be used in the next step
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
# MAGIC Data for all SKUs is logically unified within a Spark DataFrame, allowing large-scale distributed processing rather than single node procecssing with pandas. The dataframe for each key that we group by is processed in parallel

# COMMAND ----------

# MAGIC %md
# MAGIC Example of <a href="https://spark.apache.org/docs/3.2.1/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html">groupby + applyInPandas</a>:
# MAGIC ```
# MAGIC enriched_df = (
# MAGIC   spark_df
# MAGIC   .groupBy("device_id")
# MAGIC   .applyInPandas(transformation_function, expected_schema)
# MAGIC )
# MAGIC ```

# COMMAND ----------

enriched_df = (
  demand_df
  ... # TODO: group by the Product column
  ... # TODO: use applyInPandas to run our add_exo_variables function in a distributed way. We'll produce the enriched_schema we defined above
)
display(enriched_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Build, tune and score a model per each SKU with Pandas UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC Example of input and output of the function to be run via <a href="https://spark.apache.org/docs/3.2.1/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html">applyInPandas</a>:
# MAGIC ```
# MAGIC def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
# MAGIC     ... ML stuff goes here ...
# MAGIC     return return_df
# MAGIC ```
# MAGIC Example of defining a model, fitting it to data and using it to make predictions:
# MAGIC ```
# MAGIC model1 = SARIMAX(train, exog=train_exo, order=order_parameters, seasonal_order=(0, 0, 0, 0))
# MAGIC fit1 = model1.fit(disp=False)
# MAGIC fcast1 = fit1.predict(
# MAGIC             start = min(score_exo.index),
# MAGIC             end = max(score_exo.index),
# MAGIC             exog = score_exo
# MAGIC ) 
# MAGIC ```
# MAGIC Example of defining a <a href="http://hyperopt.github.io/hyperopt/getting-started/search_spaces/">search space</a>:
# MAGIC ```
# MAGIC space = {
# MAGIC   'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
# MAGIC   'd': scope.int(hyperopt.hp.quniform('d', 0, 2, 1)),
# MAGIC   'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
# MAGIC }
# MAGIC ```
# MAGIC Example of defining a <a href="http://hyperopt.github.io/hyperopt/">function to minimize</a>:
# MAGIC ```
# MAGIC argmin = fmin(             
# MAGIC   fn=evaluate_model,  
# MAGIC   space=space,            
# MAGIC   algo=tpe.suggest,
# MAGIC   max_evals=10,
# MAGIC   trials=SparkTrials(parallelism=1),
# MAGIC   rstate=rstate,
# MAGIC   verbose=False
# MAGIC )
# MAGIC ```

# COMMAND ----------

def build_tune_and_score_model(sku_pdf: ...) -> ...: # TODO: define the expected the input and output
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
          fitted_model = ...(disp=False, method='nm') # TODO: fit the model defined above to our data

          # Validation
          fcast = ...(  # TODO: define how the model should make predictions
            start=validation_data.index.min(), 
            end=validation_data.index.max(), 
            exog=validation_data[exo_fields]
          )

          return {'status': hyperopt.STATUS_OK, 'loss': np.power(validation_data.Demand.to_numpy() - fcast.to_numpy(), 2).mean()}

    search_space = {
        'p': scope.int(...('p', 0, 4, 1)), # TODO: define the distribution of the hyperopt search space
        'd': scope.int(...('d', 0, 2, 1)), # TODO: define the distribution of the hyperopt search space
        'q': scope.int(...('q', 0, 4, 1))  # TODO: define the distribution of the hyperopt search space
    }

    rstate = np.random.default_rng(123) # just for reproducibility of this notebook

    best_hparams = ...(   # TODO: provide the hyperopt function that defines what we're minimizing
      fn=...,             # TODO: provide the function that we'll use to evaluate the model (defined in line 20)
      space=...,          # TODO: provide the search space
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
# MAGIC ### Run distributed predictions

# COMMAND ----------

# MAGIC %md
# MAGIC Example of <a href="https://spark.apache.org/docs/3.2.1/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html">groupBy + applyInPandas</a>:
# MAGIC ```
# MAGIC prediction_df = (
# MAGIC   combined_df
# MAGIC   .groupby("device_id")
# MAGIC   .applyInPandas(apply_model, schema=apply_return_schema)
# MAGIC )
# MAGIC ```

# COMMAND ----------

# To maximize parallelism, we can allocate each ("Product", SKU") group its own Spark task.
# First, we'll disable Adaptive Query Execution (AQE) just for this step
spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")

# next, partition our dataframe by Product+SKU combinations
n_tasks = enriched_df.select("Product", "SKU").distinct().count()

forecast_df = (
  enriched_df
  .repartition(n_tasks, "Product", "SKU")
  ...        # TODO: group by the columns you're parallelizing by
  ...        # TODO: apply the objective function and the expected schema that we defined above
)

display(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

# MAGIC %md
# MAGIC Since we're running a lab with many people we won't write out the results, but we could by running the below cells

# COMMAND ----------

# # Write the data 
# forecast_df.write \
# .mode("overwrite") \
# .format("delta") \
# .saveAsTable(f'{db_name}.part_level_demand_with_forecasts')

# COMMAND ----------

# display(spark.sql(f"SELECT * FROM {db_name}.part_level_demand_with_forecasts"))

# COMMAND ----------

# MAGIC %md
# MAGIC You can find the full series of solution accelerator notebooks at https://github.com/databricks-industry-solutions/parts-demand-forecasting. For more information about the accelerator, visit https://www.databricks.com/solutions/accelerators/demand-forecasting.

# COMMAND ----------


