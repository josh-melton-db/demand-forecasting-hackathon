# Databricks notebook source
# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

print(cloud_storage_path)
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
from sklearn.metrics import mean_absolute_percentage_error

import pyspark.sql.functions as f
from pyspark.sql.types import *

# COMMAND ----------

demand_df = spark.read.table(f"{dbName}.part_level_demand")
demand_df = demand_df.cache() # just for this example notebook

# COMMAND ----------

example_sku = demand_df.select("SKU").orderBy("SKU").limit(1).collect()[0].SKU
print("example_sku:", example_sku)
pdf = demand_df.filter(f.col("SKU") == example_sku).toPandas()
series_df = pd.Series(pdf['Demand'].values, index=pdf['Date'])
series_df = series_df.asfreq(freq='W-MON')

forecast_horizon = ... # TODO: choose a number for the forecast horizon (in weeks). We recommend somewhere between 26-52
is_history = ... # TODO: determine the is_history column based on the FORECAST_HORIZON constant above
train = series_df.iloc[is_history]
score = series_df.iloc[~np.array(is_history)]

# COMMAND ----------

# Define the starting point for covid (a major shift in demand) and assign a Week column
covid_breakpoint = dt.date(year=2020, month=3, day=1)
exo_df = pdf.assign(Week = pd.DatetimeIndex(pdf["Date"]).isocalendar().week.tolist()) 

exo_df = exo_df \
  .assign(covid = ...) \ # TODO: assign a covid column which signifies dates greater than the covid_breakpoint
  .assign(christmas = np.where((exo_df["Week"] >= 51) & (exo_df["Week"] <= 52) , 1, 0).tolist()) \
  .assign(new_year = np.where((exo_df["Week"] >= 1) & (exo_df["Week"] <= 4)  , 1, 0).tolist()) \
  .set_index('Date')

exo_df = exo_df[["covid", "christmas", "new_year" ]]
exo_df = exo_df.asfreq(freq='W-MON')
print(exo_df) # preview our exogenous columns
train_exo = exo_df.iloc[is_history]  
score_exo = exo_df.iloc[~np.array(is_history)]

# COMMAND ----------

fit1 = SARIMAX(train, order=(1, 2, 1), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
fcast1 = fit1.predict(start = min(train.index), end = max(score_exo.index)).rename("Without exogenous variables")

fit2 = SARIMAX(train, exog=train_exo, order=(1, 2, 1), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
fcast2 = fit2.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo).rename("With exogenous variables")

# COMMAND ----------

plt.figure(figsize=(18, 6))
plt.plot(series_df, marker="o", color="black")
plt.plot(fcast1[10:], color="blue")
(line1,) = plt.plot(fcast1[10:], marker="o", color="blue")
plt.plot(fcast2[10:], color="green")
(line2,) = plt.plot(fcast2[10:], marker="o", color="green")

plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')
plt.legend([line0, line1, line2], ["Actuals", fcast1.name, fcast2.name])
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("SARIMAX")

# COMMAND ----------

# MAGIC %md
# MAGIC For the model above, we applied a manual trial-and-error method to find good parameters. MLFlow and Hyperopt can be leveraged to find optimal parameters automatically. 

# COMMAND ----------

# MAGIC %md
# MAGIC First, we must define an evaluation function. It trains a SARIMAX model with given parameters and evaluates it by calculating the mean squared error.

# COMMAND ----------

# Define an evaluation function for the SARIMAX model 
def evaluate_model(hyperopt_params):
  
  # Configure model parameters
  params = hyperopt_params
  
  assert "p" in params and "d" in params and "q" in params, "Please provide p, d, and q"
  
  if 'p' in params: params['p']=int(params['p']) # hyperopt supplies values as float but model requires int
  if 'd' in params: params['d']=int(params['d']) # hyperopt supplies values as float but model requires int
  if 'q' in params: params['q']=int(params['q']) # hyperopt supplies values as float but model requires int
    
  order_parameters = (params['p'],params['d'],params['q'])

  # For simplicity in this example, assume no seasonality
  model1 = SARIMAX(train, exog=train_exo, order=order_parameters, seasonal_order=(0, 0, 0, 0))
  fit1 = ...(disp=False) # TODO: fit model1 defined above
  fcast1 = ...(start = ..., end = ..., exog = score_exo ) # TODO: take the model which was fit to the data and make predictions starting with the minimum date and ending with the maximum date

  return {'status': hyperopt.STATUS_OK, 'loss': np.power(score.to_numpy() - fcast1.to_numpy(), 2).mean()}

# COMMAND ----------

# MAGIC %md
# MAGIC Second, we define a search space of parameters for which the model will be evaluated.

# COMMAND ----------

space = {
  'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
  'd': scope.int(hyperopt.hp.quniform('d', 0, 2, 1)),
  'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
}

# COMMAND ----------

rstate = np.random.default_rng(123)

with mlflow.start_run(run_name='mkh_test_sa'): # TODO: assign a name to our mlflow run
  argmin = ...( # TODO: specify whether we are minimizing or maximizing our evaluation metric
    fn=..., # TODO: pass the function we defined which we'll use to evaluate our model
    space=..., # TODO: pass the search space we defined above
    algo=tpe.suggest,  # this selects algorithm controlling how hyperopt navigates the search space
    max_evals=10,
    trials=SparkTrials(parallelism=1),
    rstate=rstate,
    verbose=False
    )

# COMMAND ----------

displayHTML(f"The optimal parameters for the selected series with SKU '{pdf.SKU.iloc[0]}' are: d = '{argmin.get('d')}', p = '{argmin.get('p')}' and q = '{argmin.get('q')}'")
