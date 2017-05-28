import numpy
import multiprocessing
import json
import os
import csv

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array


APP_NAME = 'IE2-Project-Homocide-Reports'
INPUT_MODEL = "C:/Users/MWeil/Documents/GitHub/IE2-Project/one-hot-clustering/kmeans_model_7/"
OUTPUT_DATA = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/one-hot_7Clusters_"

def run_kmeans(sc):
    cpu_count = multiprocessing.cpu_count()


    # Load Model
    sameModel = KMeansModel.load(sc, INPUT_MODEL)

    centers = sameModel.clusterCenters
    print("Cluster Centers: ")
    for n, center in enumerate(centers):
        out_f = OUTPUT_DATA + str(n) + "Cluster.csv"
        numpy.savetxt(out_f, center, newline=";")
        print(center)

    # Save labels in json file
    #with open(OUTPUT_LABEL, 'w') as out_f:
    #    json.dump(labels, out_f)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmeans(sc)
