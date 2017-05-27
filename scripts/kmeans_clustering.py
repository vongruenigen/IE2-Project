from __future__ import print_function

from pyspark import SparkConf, SparkContext

from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from numpy import array

import multiprocessing
import json
import math

APP_NAME = 'IE2-Project-Homocide-Reports'


"""
An example demonstrating k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/kmeans_example.py

This example requires NumPy (http://www.numpy.org/).
"""

def run_kmeans(sc):
    cpu_count = multiprocessing.cpu_count()
    cluster_loss = dict()

    # Loads data.
    for n in range(2, 10):
        dataset = sc.textFile("/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/data/samples.csv", cpu_count)
        dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))

        # Trains a k-means model.all
        clusters = KMeans.train(dataset, n)

        def error(point):
            center = clusters.centers[clusters.predict(point)]
            return math.sqrt(sum([x**2 for x in (point - center)]))

        # Evaluate clustering by computing Within Set Sum of Squared Errors.
        wssse = dataset.map(lambda x: error(x)).reduce(lambda x, y: x+y)
        print("Within Set Sum of Squared Errors = " + str(wssse))

        clusters.save(sc, "/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/results/kmeans_model_%d" % n)
        cluster_loss[n] = wssse

    # Write Cluster loss pairs into json file
    with open('/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/results/kmeans_elbowCurveData.json', 'w') as out_f:
        json.dump(cluster_loss, out_f)


if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmeans(sc)
