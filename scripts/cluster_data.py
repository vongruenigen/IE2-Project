from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array

import multiprocessing
import json
import os

APP_NAME = 'IE2-Project-Homocide-Reports'
INPUT_DATA = "C:/Users/MWeil/Documents/GitHub/IE2-Project/emb/emb-out.txt"
INPUT_LABEL = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/database_new_label.csv"
CLUSTER_OUT_PATH = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/clusters/cluster_"
CLUSTERS = 7

def run_kmeans(sc):
    cpu_count = multiprocessing.cpu_count()

    # Load Data
    dataset = sc.textFile(INPUT_DATA, cpu_count)
    #dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))

    labelset = sc.textFile(INPUT_LABEL, cpu_count)
    labelset = labelset.map(lambda line: array([float(x) for x in line.split(', ')]))
#
    labels = labelset.cache().take(1)[0]
    for n,label in enumerate(labels):
        filename = CLUSTER_OUT_PATH + "%d.csv" % label
        line = (dataset.zipWithIndex()
            .filter(lambda x: x[1] == n+1)
            .map(lambda x: x[0])
            .collect())
        file = open(filename, 'a+')
        file.write(''.join(line) + "\n")
        file.close


if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmeans(sc)
