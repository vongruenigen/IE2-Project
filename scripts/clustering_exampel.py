#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import json

"""
An example demonstrating k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/kmeans_example.py

This example requires NumPy (http://www.numpy.org/).
"""

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("KMeansExample")\
        .getOrCreate()

    cluster_loss = dict()

    # Loads data.
    for n in range(2, 10):
        dataset = spark.read.csv("C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/database.txt", sep= " ", inferSchema=True)
        dataVector = VectorAssembler(inputCols=dataset.columns[0:], outputCol='features')
        output = dataVector.transform(dataset)
        # Trains a k-means model.
        kmeans = KMeans().setK(n).setSeed(1)
        model = kmeans.fit(output)

        # Evaluate clustering by computing Within Set Sum of Squared Errors.
        wssse = model.computeCost(output)
        print("Within Set Sum of Squared Errors = " + str(wssse))

        # Shows the result.
        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)
        cluster_loss[n] = wssse
    # Write Cluster loss pairs into json file
    with open('C:/Users/MWeil/Documents/GitHub/IE2-Project/results/kmeans_elbowCurveData.json', 'w') as out_f:
        json.dump(cluster_loss, out_f)
    import pdb; pdb.set_trace()
    spark.stop()
