import os
import sys
import time
import numpy as np

from collections import defaultdict
from operator import itemgetter
from os import path

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: process_data.py <data-csv> <out-csv>')
    sys.exit(2)

RAW_DATA_PATH = argv[0]
PREPROCESSED_DATA_PATH = argv[1]

# Columns we are going to ignore in the dataset
# because they're redundant or non-informative
STRIP_COLS = ('Record ID', 'Agency Name', 'Agency Code',
              'Year', 'Month', 'Record Source')

if path.isfile(PREPROCESSED_DATA_PATH):
    msg = 'Output file at "%s" already exists, delete? (y/n): ' % PREPROCESSED_DATA_PATH

    if input(msg).strip('\n').lower() == 'y':
        os.remove(PREPROCESSED_DATA_PATH)
    else:
        print('Quitting')
        sys.exit(2)

with open(data_path, 'r') as data_f:
    with open(out_path, 'w+') as out_f:
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

        sum_lines = last_idx+1
        sum_vec_entries = 0

        for c, v in col_values.items():
            print('  %s = %d' % (c, len(v)))
            sum_vec_entries += len(v)

        print('\nThe generated vectors will have a total of %d entries each' % sum_vec_entries)
        print('The dataset has %i samples' % sum_lines)

        data_f.seek(0)
        data_f.readline() # skip headings after seek(0)

        start_time = time.time()
        curr_idx = 0
        temp_x = []

        for i, line in enumerate(data_f):
            sample_values = line.strip('\n').split(',')
            sample_vec = np.zeros(sum_vec_entries)
            idx_offset = 0

            for c, v in zip(columns, sample_values):
                if c in STRIP_COLS: continue
                sample_vec[col_values[c].index(v)+idx_offset] = 1
                idx_offset += len(col_values[c])

            temp_x.append(sample_vec)

            if (i+1) % 100000 == 0 or (i+1) == sum_lines:
                temp_x = np.array(temp_x)
                np.random.shuffle(temp_x)

                print('Processed %i samples (%.1f%%)...' % (i+1, 100*(float(i+1)/sum_lines)))
                print('Storing collected data in CSV file...')

                temp_x_str = []

                for i in range(temp_x.shape[0]):
                    temp_x_str.append(';'.join(map(str, map(int, temp_x[i]))))
           
                out_f.write('%s\n' % '\n'.join(temp_x_str))

                curr_idx += temp_x.shape[0]

                print('Stored data successfully! (Took %.2fs)' % (time.time() - start_time))
                start_time = time.time()
                temp_x = []

        print('Successfully stored preprocessed samples in: %s' % PREPROCESSED_DATA_PATH)
