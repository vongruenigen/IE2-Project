#!/usr/bin/env python

import os
import re
import sys
import tensorflow as tf
import h5py
import time

from os import path
from numpy import random

# Some mangling with the load path
source_path = path.realpath('./source')
sys.path.insert(0, source_path)

from autoencoders import AutoEncoder, VariationalAutoencoder

TRAINING_EPOCHS = 1000
BATCH_SIZE = 128
DISPLAY_EPOCH = 1
DISPLAY_BATCH = 1000
HIDDEN_SIZE = 256
RESULTS_DIR = path.abspath(path.join(path.dirname(__file__), 'results'))

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
    log('ERROR: ./run.py <samples.py> [<mode=(train|embed)> <model-dir> <emb-out.h5>]')
    sys.exit(2)

if len(argv) > 3:
    mode = argv[1]
    emb_out_path = argv[2]

encoder_type = camel_to_sneak(CurrentAutoEncoder.__name__)
time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
result_name = '%s-%s-results/' % (time_stamp, encoder_type)
result_path = path.join(RESULTS_DIR, result_name)

if not path.isdir(result_path):
    os.mkdir(result_path)
else:
    error('Result directory "%s" already exists' % result_path)
    sys.exit(2)

def get_next_batch(train_data, num_batch, random=False, available_idxs=None):
    idxs = None

    if random:
        idxs = [available_idxs.pop(random.randint(0, len(available_idxs))) \
                for _ in range(BATCH_SIZE)]
        idxs = list(sorted(idxs))
    else:
        start_idx = num_batch*BATCH_SIZE
        idxs = range(start_idx, start_idx+BATCH_SIZE)

    batch_data = train_data[idxs]

    if random:
        return (batch_data, available_idxs)
    else:
        return batch_data

with tf.Session() as session:
    if mode == 'train':
        train_data_path = argv[0]
        loss_track = []

        with h5py.File(train_data_path) as train_f:
            train_data = train_f['x']
            input_size = train_data.shape[1]
            num_samples = train_data.shape[0]

            log('Starting training with a %s' % CurrentAutoEncoder.__name__)

            autoencoder = CurrentAutoEncoder(input_size, HIDDEN_SIZE, session=session)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

            session.run(tf.global_variables_initializer())

            for epoch in range(TRAINING_EPOCHS):
                log('Starting epoch #%d' % (epoch+1))
                num_batches = int(num_samples / BATCH_SIZE)
                avg_loss = 0

                for num_batch in range(num_batches):
                    batch_x = get_next_batch(train_data, num_batch)
                    loss = autoencoder.batch_fit(batch_x)
                    avg_loss += (loss / num_samples) * BATCH_SIZE

                    if (num_batch+1) % DISPLAY_BATCH == 0 or (num_batches-num_batch) < 5:
                        log('Batch #%d of #%d, loss = %.5f' % (num_batch+1, num_batches, loss))

                if (epoch+1) % DISPLAY_EPOCH == 0 or (epoch+1) == TRAINING_EPOCHS:
                    log('Epoch #%d of #%d, loss = %.5f' % (epoch+1, TRAINING_EPOCHS, avg_loss))
                    saver.save(session, result_path)

    elif mode == 'embed':
        samples_path = argv[0]

        with h5py.File(samples_path) as samples_f:
            with h5py.File(emb_out_path) as emb_f:
                samples_data = samples_f['x']
                num_samples = samples_data.shape[0]
                samples_embs = emb_f.create_dataset('y', dtype='float32',
                                                    shape=(num_samples, HIDDEN_SIZE))

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

                    start_idx = num_batch * BATCH_SIZE
                    end_idx = start_idx + BATCH_SIZE

                    samples_embs[start_idx:end_idx] = batch_y

                    if (i+1) % DISPLAY_BATCH == 0:
                        log('Processed %d of %d samples' % (i+1, sum_lines))
    else:
        error('Invalid mode %s' % mode)
