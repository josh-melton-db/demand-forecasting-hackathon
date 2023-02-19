# Databricks notebook source
from pyspark.sql.functions import sum as sqlsum
from graphframes.lib import AggregateMessages as AM
from graphframes import GraphFrame

# COMMAND ----------

# Create a Vertex DataFrame with unique ID column "id"
v = spark.createDataFrame([
  ("a", "Alice", 36),
  ("b", "Bob", 30),
  ("c", "Charlie", 25),
], ["id", "name", "age"])
# Create an Edge DataFrame with "src" and "dst" columns
e = spark.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
], ["src", "dst", "relationship"])

# Create a GraphFrame
g = GraphFrame(v, e)

# COMMAND ----------

# Count the number of "follow" connections in the graph.
num_edges = g.edges.filter("relationship = 'follow'").count()
print("Follow edges:", num_edges)

# COMMAND ----------

# Query: Get in-degree of each vertex.
g.inDegrees.display()

# COMMAND ----------

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.01, maxIter=1)
results.vertices.select("id", "pagerank").display()

# COMMAND ----------

# For each user, sum the ages of the adjacent users.
msgToSrc = AM.dst["age"]
msgToDst = AM.src["age"]

# COMMAND ----------

agg_connected = g.aggregateMessages(
  sqlsum(AM.msg).alias("summedConnectedAges"),
  sendToSrc=msgToSrc,
  sendToDst=msgToDst)
agg_connected.orderBy('summedConnectedAges').display()

# COMMAND ----------

agg_to_src = g.aggregateMessages(
  sqlsum(AM.msg).alias("summedSrcAges"),
  sendToSrc=msgToSrc,
  sendToDst=None)
agg_to_src.orderBy('summedSrcAges').display()

# COMMAND ----------

agg_to_src = g.aggregateMessages(
  sqlsum(AM.msg).alias("summedDstAges"),
  sendToSrc=None,
  sendToDst=msgToDst)
agg_to_src.orderBy('summedDstAges').display()

# COMMAND ----------


