from pyspark import SparkConf, SparkContext
from pyspark_kmodes import EnsembleKModes
from pyspark.sql import SparkSession
from numpy import array
from collections import defaultdict

import multiprocessing
import json
import math

APP_NAME = 'IE2-Project-Homocide-Reports'

def run_kmodes(sc):
    cpu_count = multiprocessing.cpu_count()
    cluster_infos = {}

    # Loads data.
    for n in range(2, 10):
        dataset = sc.textFile("/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/data/samples.csv", cpu_count)
        dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))

        method = EnsembleKModes(n, 20)
        model = method.fit(dataset)

        cluster_infos['cost'] = model.mean_cost
        cluster_infos['centroids'] = model.clusters

        # Write Cluster loss pairs into json file
        with open('/media/dvg/Volume/Dropbox/ZHAW/IE2/Project/results/kmodes_%d_results.json' % n, 'w') as out_f:
            json.dump(cluster_infos, out_f)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmodes(sc)
