#!/usr/bin/env python

import os
import re
import sys
import tensorflow as tf
import numpy as np
import time

from os import path
from numpy import random

# Some mangling with the load path
source_path = path.realpath('./source')
sys.path.insert(0, source_path)

from autoencoders import AutoEncoder, VariationalAutoencoder

TRAINING_EPOCHS = 500
BATCH_SIZE = 128
DISPLAY_EPOCH = 1
DISPLAY_BATCH = 1000
HIDDEN_SIZE = 256
RESULTS_DIR = path.abspath(path.join(path.dirname(__file__), 'results'))

#Autoencoder Klasse hier angeben, alternativ VariationalAutoencoder
CurrentAutoEncoder = AutoEncoder

def camel_to_sneak(name):
    '''Convert a string from camel-case to sneak-case.'''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def error(msg):
    log(msg, level='error')

def log(msg, level='info'):
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('[%s][%s] %s' % (level.upper(), ts, msg))
    if level == 'error': sys.exit(2)

mode = 'train'
emb_out_path = None
model_dir_path = None

argv = sys.argv[1:]

if len(argv) == 0:
    log('ERROR: ./run.py <samples-csv> [<mode=(train|embed)> <model-dir> <emb-out-csv>]')
    sys.exit(2)

if len(argv) > 3:
    mode = argv[1]
    model_path = argv[2]
    emb_out_path = argv[3]

encoder_type = camel_to_sneak(CurrentAutoEncoder.__name__)
time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
result_name = '%s-%s-results/' % (time_stamp, encoder_type)
result_path = path.join(RESULTS_DIR, result_name)

if not path.isdir(result_path):
    os.makedirs(result_path)
else:
    error('Result directory "%s" already exists' % result_path)
    sys.exit(2)

def get_next_batch(train_data, batch_size):
    new_batch = []

    for _ in range(batch_size):
        new_line = train_data.readline().strip('\n')

        if new_line == '':
            continue

        new_batch.append(np.array(list((map(float, new_line.split(';'))))))

    return np.stack(new_batch)

def get_input_size_and_length(data_f):
    input_size = len(data_f.readline().split(';'))
    data_f.seek(0)

    num_samples = sum([1 for _ in data_f])
    data_f.seek(0)

    return input_size, num_samples

with tf.Session() as session:
    if mode == 'train':
        train_data_path = argv[0]
        loss_track = []

        with open(train_data_path, 'r') as train_f:
            input_size, num_samples = get_input_size_and_length(train_f)

            log('Starting training with a %s' % CurrentAutoEncoder.__name__)

            autoencoder = CurrentAutoEncoder(input_size, HIDDEN_SIZE, session=session)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

            session.run(tf.global_variables_initializer())

            for epoch in range(TRAINING_EPOCHS):
                log('Starting epoch #%d' % (epoch+1))
                num_batches = int(num_samples / BATCH_SIZE)
                avg_loss = 0

                for num_batch in range(num_batches):
                    batch_x = get_next_batch(train_f, BATCH_SIZE)
                    loss = autoencoder.batch_fit(batch_x)
                    avg_loss += (loss / num_samples) * BATCH_SIZE

                    if (num_batch+1) % DISPLAY_BATCH == 0 or (num_batches-num_batch) < 5:
                        log('Batch #%d of #%d, loss = %.5f' % (num_batch+1, num_batches, loss))

                if (epoch+1) % DISPLAY_EPOCH == 0 or (epoch+1) == TRAINING_EPOCHS:
                    log('Epoch #%d of #%d, loss = %.5f' % (epoch+1, TRAINING_EPOCHS, avg_loss))
                    saver.save(session, result_path)

    elif mode == 'embed':
        samples_path = argv[0]

        with open(samples_path, 'r') as samples_f:
            with open(emb_out_path, 'w+') as emb_f:
                input_size, num_samples = get_input_size_and_length(samples_f)
                log('Restoring model from %s' % model_path)

                autoencoder = CurrentAutoEncoder(input_size, HIDDEN_SIZE, session=session)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
                saver.restore(session, model_path)

                log('Finished restoring the model')
                log('Starting to embed the samples in %s' % samples_path)

                num_batches = int(num_samples / BATCH_SIZE)

                for num_batch in range(num_batches):
                    batch_x = get_next_batch(samples_data, num_batch)
                    batch_y = autoencoder.transform(batch_x)

                    for y in batch_y:
                        emb_f.write('%s\n' % ';'.join(map(float, y)))

                    if (num_batch+1) % DISPLAY_BATCH == 0:
                        log('Processed %d of %d samples' % (num_batch+1, num_batches))
    else:
        error('Invalid mode %s' % mode)
