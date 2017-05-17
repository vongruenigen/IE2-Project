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

TRAINING_EPOCHS = 20
BATCH_SIZE = 128
DISPLAY_EPOCH = 1
DISPLAY_BATCH = 1000
HIDDEN_SIZE = 512
RESULTS_DIR = path.abspath(path.join(path.dirname(__file__), 'results'))

CurrentAutoEncoder = AutoEncoder

mode = 'train'
argv = sys.argv[1:]

if len(argv) < 1:
    log('ERROR: ./run.py <samples.py>')
    sys.exit(2)

def camel_to_sneak(name):
    '''Convert a string from camel-case to sneak-case.'''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def log(msg, level='info'):
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print('[%s][%s] %s' % (level.upper(), ts, msg))

encoder_type = camel_to_sneak(CurrentAutoEncoder.__name__)
time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
result_name = '%s-%s-results' % (time_stamp, encoder_type)
result_path = path.join(RESULTS_DIR, result_name)

if not path.isdir(result_path):
    os.mkdir(result_path)
else:
    log('ERROR: Result directory "%s" already exists' % result_path)
    sys.exit(2)

def get_next_batch(train_data, epoch, random=False, available_idxs=None):
    # Gather random indices we want to use for this batch
    idxs = None

    if random:
        idxs = [available_idxs.pop(random.randint(0, len(available_idxs))) \
                for _ in range(BATCH_SIZE)]
        idxs = list(sorted(idxs))
    else:
        start_idx = epoch*BATCH_SIZE
        idxs = range(start_idx, start_idx+BATCH_SIZE)

    batch_data = train_data[idxs]

    if random:
        return (batch_data, available_idxs)
    else:
        return batch_data

with tf.Session() as session:
    if mode == 'train':
        train_data_path = argv[0]

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
                avg_loss = 0
                num_batches = int(num_samples / BATCH_SIZE)

                for num_batch in range(num_batches):
                    batch_x = get_next_batch(train_data, epoch)

                    loss = autoencoder.batch_fit(batch_x)
                    avg_loss += (loss / num_samples) * BATCH_SIZE

                    if (num_batch+1) % DISPLAY_BATCH == 0 or (num_batches - num_batch) < 5:
                        log('Batch #%d of #%d, loss = %.5f' % (num_batch+1, num_batches, loss))

                if (epoch+1) % DISPLAY_EPOCH == 0:
                    log('Epoch #%d of #%d, loss = %.5f' % (epoch+1, TRAINING_EPOCHS, avg_loss))
                    saver.save(session, result_path)
    else:
        log('GENERATING EMBEDDINGS NOT IMPLEMENTED YET!')
