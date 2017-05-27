from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array

import multiprocessing
import json
import os

APP_NAME = 'IE2-Project-Homocide-Reports'
INPUT_DATA = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/database_new.csv"
OUTPUT_LABEL = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/database_new_label.json"
INPUT_MODEL = "C:/Users/MWeil/Documents/GitHub/IE2-Project/one-hot-clustering/kmeans_model_7"

def run_kmeans(sc):
    cpu_count = multiprocessing.cpu_count()

    # Load Data
    dataset = sc.textFile(INPUT_DATA, cpu_count)
    dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))

    # Load Model
    sameModel = KMeansModel.load(sc, INPUT_MODEL)

    # Predict cluster labels per row
    labels = sameModel.predict(dataset).collect()

    # Save labels in json file
    with open(OUTPUT_LABEL, 'w') as out_f:
        json.dump(labels, out_f)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_kmeans(sc)
