import os
import sys
import h5py
import numpy as np

from collections import defaultdict

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: process_data.py <data-csv> <out-npy>')
    sys.exit(2)

data_path = argv[0]
out_path = argv[1]

# Columns we are going to ignore in the dataset
# because they're redundant or non-informative
STRIP_COLS = ('Record ID', 'Agency Name', 'Agency Code', 'Year', 'Month', 'Record Source')

with open(data_path, 'r') as data_f:
    with h5py.File(out_path) as out_f:
        # Read headings
        columns = data_f.readline().strip('\n').split(',')
        col_values = defaultdict(list)
        last_idx = 0

        print('The following columns are filtered: %s' % ', '.join(STRIP_COLS))

        # Find all unique values for each row in the dataset
        # and store them in col_values.
        for i, line in enumerate(data_f):
            sample_values = line.strip('\n').split(',')

            for c, v in zip(columns, sample_values):
                if c in STRIP_COLS: continue
                if v not in col_values[c]: col_values[c].append(v)

            last_idx = i

        print('The number of distinct values for each column are:\n')

        sum_lines = last_idx + 1
        sum_vec_entries = 0

        for c, v in col_values.items():
            print('  %s = %d' % (c, len(v)))
            sum_vec_entries += len(v)

        print('\nThe generated vectors will have a total of %d entries each' % sum_vec_entries)
        print('The dataset has %i samples\n' % sum_lines)

        data_f.seek(0)
        data_f.readline() # skip headings after seek(0)

        X = out_f.create_dataset('x', dtype='i8', shape=(sum_lines, sum_vec_entries))

        for i, line in enumerate(data_f):
            sample_values = line.strip('\n').split(',')

            sample_vec = np.zeros(sum_vec_entries)
            idx_offset = 0

            for c, v in zip(columns, sample_values):
                if c in STRIP_COLS: continue
                sample_vec[col_values[c].index(v)+idx_offset] = 1
                idx_offset += len(col_values[c])

            X[i] = sample_vec

            if (i+1) % 10000 == 0:
                print('Processed %.1f%% of the samples...' % (100*(float(i+1)/sum_lines)))

        print('Successfully stored preprocessed samples in: %s' % out_path)
