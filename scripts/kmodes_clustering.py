from pyspark import SparkConf, SparkContext
from pyspark_modes import EnsembleKModes
from pyspark.sql import SparkSession
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

def run_kmodes(sc):
    cpu_count = multiprocessing.cpu_count()
    cluster_infos = {}

    # Loads data.
    for n in range(2, 10):
        dataset = sc.textFile("/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/data/samples.csv")
        dataset = sc.parallelize(dataset, cpu_count)
        dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))

        method = EnsembleKModes(n, 50)
        model = method.fit(dataset)

        cluster_infos[n]['cost'] = model.mean_cost
        cluster_infos[n]['centroids'] = model.clusters

    # Write Cluster loss pairs into json file
    with open('/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/results/kmodes_results.json', 'w') as out_f:
        json.dump(cluster_infos, out_f)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmodes(sc)
