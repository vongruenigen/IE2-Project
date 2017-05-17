#!/usr/bin/env python

import os
import sys
import tensorflow as tf
import h5py

from os import path

# Some mangling with the load path
source_path = path.realpath('./source')
sys.path.insert(0, source_path)

from variational_autoencoder import VariationalAutoencoder

TRAINING_EPOCHS = 20
BATCH_SIZE = 128
DISPLAY_EPOCH = 1
DISPLAY_BATCH = 100
HIDDEN_SIZE = 512

mode = 'train'
argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: ./run.py <samples.py> <model-out>')
    sys.exit(2)

def get_next_batch(train_data, epoch):
    start_idx = epoch * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE

    return train_data[start_idx:end_idx]

with tf.Session() as session:
    if mode == 'train':
        train_data_path = argv[0]
        model_out_path = argv[1]

        with h5py.File(train_data_path) as train_f:
            train_data = train_f['x']
            input_size = train_data.shape[1]
            num_samples = train_data.shape[0]

            autoencoder = VariationalAutoencoder(input_size, HIDDEN_SIZE, session=session)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

            session.run(tf.global_variables_initializer())

            for epoch in range(TRAINING_EPOCHS):
                avg_loss = 0
                num_batches = int(num_samples / BATCH_SIZE)

                for num_batch in range(num_batches):
                    batch_x = get_next_batch(train_data, epoch)
                    loss = autoencoder.batch_fit(batch_x)
                    avg_loss += (loss / num_samples) * BATCH_SIZE

                    if (num_batch+1) % DISPLAY_BATCH == 0:
                        print('Batch #%d of #%d, loss = %.5f' % (num_batch+1, num_batches, avg_loss))

                if (epoch+1) % DISPLAY_EPOCH == 0:
                    print('Epoch #%d of #%d, loss = %.5f' % (epoch+1, TRAINING_EPOCHS, avg_loss))
                    saver.save(session, model_out_path)
    else:
        print('NOT IMPLEMENTED YET!')
