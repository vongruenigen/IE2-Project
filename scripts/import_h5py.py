import h5py
import sys
import os

argv = sys.argv[1😏
file = argv[0]
out_file = argv[1]

with h5py.File(file, 'r') as in_f:
    dataset = in_f['x']

    with open(out_file, 'w+') as out_f:
        vecs_to_write = []
        max_len = len(dataset)

        for i, vec in enumerate(dataset):
            vecs_to_write.append(vec)

            if len(vec) == 10**5 or (i+1) == max_len:
                for v in vecs_to_write:
                    out_f.write('%s\n' % ' '.join(map(str, v)))
                print('Written %d vectors to txt file...' % (i+1))
                vecs_to_write = [