-------------------------------------------spark architecture-------------------------------------


from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, sum, count, max, min, countDistinct, when, lit, round,
    rank, dense_rank, row_number, desc, asc, year, month, to_date
)
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Employee PySpark Demo") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()


# 2. Sample Employee Data
data = [
    ("E001", "Alice",   "HR",       "Female",  32000, "2020-03-15",  8, "New York"),
    ("E002", "Bob",     "IT",       "Male",    48000, "2019-06-10",  6, "Chicago"),
    ("E003", "Carol",   "Finance",  "Female",  52000, "2018-01-22", 10, "Boston"),
    ("E004", "David",   "IT",       "Male",    45000, "2021-09-05",  4, "Seattle"),
    ("E005", "Emma",    "HR",       "Female",  31000, "2022-04-18",  3, "New York"),
  
]

# 3. Define Schema
schema = StructType([
    StructField("emp_id",    StringType(), False),
    StructField("name",      StringType(), False),
    StructField("department",StringType(), True),
    StructField("gender",    StringType(), True),
    StructField("salary",    IntegerType(), False),
    StructField("join_date", StringType(), False),  # we'll cast to Date
    StructField("experience",IntegerType(), True),
    StructField("city",      StringType(), True)
])

# 4. Create DataFrame
df = spark.createDataFrame(data, schema)

# 5. Cast join_date to proper Date type
df = df.withColumn("join_date", to_date(col("join_date"), "yyyy-MM-dd"))

# 6. Basic DataFrame operations
print("=== Original DataFrame ===")
df.show(12, truncate=False)
df.printSchema()

# 7. Select & alias
df.select(
    col("emp_id").alias("ID"),
    "name",
    col("salary").alias("Annual_Salary")
).show(5)

# 8. Filter examples
print("\nEmployees with salary > 45000:")
df.filter(col("salary") > 45000).show()

print("\nIT department females:")
df.filter((col("department") == "IT") & (col("gender") == "Female")).show()

print("\nMissing gender or experience < 5:")
df.filter(col("gender").isNull() | (col("experience") < 5)).show()

# 9. WithColumn - derived columns
df_enriched = df.withColumn("bonus", round(col("salary") * 0.12, 0)) \
                .withColumn("total_comp", col("salary") + col("bonus")) \
                .withColumn("senior", when(col("experience") >= 10, "Senior").otherwise("Mid/Junior"))

# 10. Aggregations - groupBy + agg
print("\n=== Department wise statistics ===")
df_enriched.groupBy("department").agg(
    count("*").alias("headcount"),
    round(avg("salary"), 0).alias("avg_salary"),
    round(sum("salary"), 0).alias("total_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary"),
    countDistinct("city").alias("unique_cities")
).orderBy(desc("avg_salary")).show()

# 11. More aggregate functions
print("\nOverall stats:")
df.agg(
    count("*").alias("total_employees"),
    countDistinct("department").alias("dept_count"),
    round(avg("salary"), 0).alias("company_avg_salary"),
    round(sum("salary") / 1000, 1).alias("total_salary_k")
).show()

# 12. Window functions - ranking within department
window_spec = Window.partitionBy("department").orderBy(desc("salary"))

df_ranked = df.withColumn("rank", rank().over(window_spec)) \
              .withColumn("dense_rank", dense_rank().over(window_spec)) \
              .withColumn("row_num", row_number().over(window_spec))

print("\nTop earners per department:")
df_ranked.filter(col("rank") <= 2).select(
    "department", "name", "salary", "rank", "dense_rank", "row_num"
).orderBy("department", "rank").show()

# 13. Convert to RDD and back
rdd = df.rdd

# Example RDD operations
print("\nRDD - employees from New York:")
ny_employees = rdd.filter(lambda x: x.city == "New York") \
                  .map(lambda x: (x.name, x.salary)) \
                  .collect()
print(ny_employees)

# 14. RDD → DataFrame again
df_from_rdd = spark.createDataFrame(rdd, schema)
print("\nDataFrame recovered from RDD:")
df_from_rdd.select("name", "department", "salary").show(5)

# 15. SQL style
df.createOrReplaceTempView("employees")

print("\nSQL - Average salary by gender and department:")
spark.sql("""
    SELECT department, gender, 
           ROUND(AVG(salary), 0) as avg_salary,
           COUNT(*) as count
    FROM employees
    WHERE gender IS NOT NULL
    GROUP BY department, gender
    ORDER BY department, avg_salary DESC
""").show()

# 16. Final cleanup (optional)
spark.stop()

print("\n--- End of PySpark Employee DataFrame Demo ---")
