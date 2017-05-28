from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from operator import itemgetter

import multiprocessing
import json
import os
import sys

APP_NAME = 'IE2-Project-Homocide-Reports'
INPUT_DATA = "/Users/dvg/Dropbox/ZHAW/IE2/Project/data/emb-out.csv"
INPUT_LABEL = "/Users/dvg/Dropbox/ZHAW/IE2/Project/results/database_new_label.csv"
CLUSTER_OUT_PATH = "/Users/dvg/Dropbox/ZHAW/IE2/Project/results/cluster_result"
CLUSTERS = 7

def custom_zip(rdd1, rdd2):
    def prepare(rdd, npart):
        return (rdd.zipWithIndex().sortBy(itemgetter(1), numPartitions=npart).keys())

    npart = rdd1.getNumPartitions() + rdd2.getNumPartitions() 
    return prepare(rdd1, npart).zip(prepare(rdd2, npart)) 

def convert_to_csv(x):
    return ';'.join(map(str, [x[0]] + x[1].tolist()))

def cluster_data(sc):
    cpu_count = multiprocessing.cpu_count()

    # labelset = sc.textFile(INPUT_LABEL, cpu_count)
    # labelset = labelset.flatMap(lambda line: [int(x) for x in line.split(',')])
    labelset = None

    with open(INPUT_LABEL, 'r') as f:
        line = f.readline()
        labelset = sc.parallelize([int(x) for x in line.split(',')])

    # Load Data
    dataset = sc.textFile(INPUT_DATA, cpu_count)
    dataset = dataset.map(lambda line: array([float(x) for x in line.split(';')]))
    
    all_ds = custom_zip(labelset, dataset).map(convert_to_csv)
    all_ds.coalesce(1).saveAsTextFile('%s_all' % CLUSTER_OUT_PATH)

    for n in range(0, CLUSTERS):
        cluster_ds = custom_zip(labelset, dataset).filter(lambda x: x[0] == n)
        cluster_ds = cluster_ds.map(convert_to_csv)
        cluster_ds.coalesce(1).saveAsTextFile('%s_%d' % CLUSTER_OUT_PATH)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    cluster_data(sc)
