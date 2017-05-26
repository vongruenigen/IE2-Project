import os
import sys

# Configure the environment
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = "C:/Users/MWeil/Spark/spark-2.1.1-bin-hadoop2.7"

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

# Add the PySpark/py4j to the Python Path
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "lib"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))


import pyspark.ml.clustering.KMeans
from pyspark.ml.feature import VectorAssembler

#Load Data
data = spark.read.csv("/FileStore/tables/vaxde2ax1495711899146/database.csv")

# Use VectorAssembler to specify features
dataVector = VectorAssembler(inputCols=data.columns[0:], outputCol='features')
output = dataVector.transform(data)

#Train a k-means model with 2 clusters
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(output)

#Evaluate clustering by computing within Set Sum of Squared Errors
wssse = model.computeCost(output)
print("Within Set Sum of Squared Error = " + str(wssse))
#textFile = sc.textFile("/FileStore/tables/vaxde2ax1495711899146/database.csv")
#data.take(1)