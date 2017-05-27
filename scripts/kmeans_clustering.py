
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans
from numpy import array

import multiprocessing
import json
import math

APP_NAME = 'IE2-Project-Homocide-Reports'

def run_kmeans(sc):
    cpu_count = multiprocessing.cpu_count()
    cluster_loss = dict()

    # Loads data.
    for n in range(2, 10):
        dataset = sc.textFile(INPUT_DATA, cpu_count)
        dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))

        # Trains a k-means model.all
        clusters = KMeans.train(dataset, n)

        def error(point):
            center = clusters.centers[clusters.predict(point)]
            return math.sqrt(sum([x**2 for x in (point - center)]))

        # Evaluate clustering by computing Within Set Sum of Squared Errors.
        wssse = dataset.map(lambda x: error(x)).reduce(lambda x, y: x+y)
        print("Within Set Sum of Squared Errors = " + str(wssse))

        clusters.save(sc, OUTPUT_MODEL_PATH + "_%d" % n)
        cluster_loss[n] = wssse

    # Write Cluster loss pairs into json file
    with open( OUTPUT_KMEANS_CURVE, 'w') as out_f:
        json.dump(cluster_loss, out_f)


if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmeans(sc)
