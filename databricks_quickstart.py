# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Quickstart
# MAGIC 
# MAGIC This notebook demonstrates basic Spark usage:
# MAGIC - Inspect Spark
# MAGIC - Create a DataFrame
# MAGIC - Run SQL
# MAGIC - Use widgets

# COMMAND ----------
# If running outside Databricks, initialize Spark
try:
    _ = spark  # noqa: F821 - provided by Databricks
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("Quickstart").getOrCreate()

print(f"Spark version: {spark.version}")

# COMMAND ----------
# Create a simple DataFrame
from pyspark.sql import functions as F

sample_rows = [
    ("Alice", 34),
    ("Bob", 23),
    ("Carol", 45),
    ("Dave", 29),
]
people_df = spark.createDataFrame(sample_rows, ["name", "age"])  # type: ignore[name-defined]

# Display the DataFrame (Databricks) or show() elsewhere
try:
    display(people_df)  # type: ignore[name-defined]
except NameError:
    people_df.show(truncate=False)

people_df.createOrReplaceTempView("people")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT * FROM people ORDER BY age DESC

# COMMAND ----------
# Widgets example: filter by minimum age
min_age_default = "30"

# Detect whether running in Databricks
running_in_databricks = False
try:
    _ = dbutils  # type: ignore[name-defined]
    running_in_databricks = True
except NameError:
    running_in_databricks = False

# Create widget if available
if running_in_databricks:
    try:
        dbutils.widgets.text("min_age", min_age_default, "Minimum Age")  # type: ignore[name-defined]
    except Exception:
        pass

# Read widget value (or fallback)
min_age_str = min_age_default
if running_in_databricks:
    try:
        min_age_str = dbutils.widgets.get("min_age")  # type: ignore[name-defined]
    except Exception:
        pass

try:
    min_age = int(min_age_str)
except ValueError:
    min_age = int(min_age_default)

filtered_df = spark.sql(f"SELECT * FROM people WHERE age >= {min_age} ORDER BY age")  # type: ignore[name-defined]
try:
    display(filtered_df)  # type: ignore[name-defined]
except NameError:
    filtered_df.show(truncate=False)

# COMMAND ----------
# Cleanup widgets (Databricks only)
if running_in_databricks:
    try:
        dbutils.widgets.remove("min_age")  # type: ignore[name-defined]
    except Exception:
        pass