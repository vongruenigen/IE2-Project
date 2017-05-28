from __future__ import print_function

from pyspark import SparkConf, SparkContext

from pyspark.mllib.feature import PCA
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession
from numpy import array

import multiprocessing
import json
import math

APP_NAME = 'IE2-Project-Homocide-Reports'
CLUSTERS = 7
CLUSTER_PATH = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/homicide-reports/cluster_result_onehot/"
PCA_PATH = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/pca/"
def run_pca(sc):
    cpu_count = multiprocessing.cpu_count()
    cluster_loss = dict()

    for n in range(0, CLUSTERS):
        filename = "cluster_" + str(n) + ".csv"
        cl_file = CLUSTER_PATH + filename
        dataset = sc.textFile(cl_file, cpu_count)
        dataset = dataset.map(lambda line: Vectors.dense([float(x) for x in line.split(';')]))

        model = PCA(2).fit(dataset)
        transformed = model.transform(dataset)
        transformed_csv = transformed.map(lambda x: ';'.join(list(map(str, x))))
        transformed_csv.coalesce(1).saveAsTextFile(PCA_PATH + "onehot_%s" % filename)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    run_pca(sc)
