from collections import namedtuple
from enum import Enum
import numpy as np
import multiprocessing as mp
from SAW import SAW
import h5py

point = namedtuple('point', ['x','y'])


SAWSample = namedtuple('SAWSample', \
        ['px','length','history'])

#method 1 of computing px
def draw_SAW_samples1(batch):
    saw = SAW(10)
    px = 1
    attempts = 0
    samples = []
    fin_samples = []
   
    for b in range(batch):
        saw.reset()
        attempts += 1
        while saw.getNumValidMoves() > 0:
            px *= saw.getNumValidMoves()
            saw.randomMove()
            if saw.pos == point(10,10):
                fin_samples.append(SAWSample(px/attempts, saw.length,saw.history))
        samples.append(SAWSample(px,saw.length, saw.history))
    return samples, fin_samples
#method 1 of computing px
def draw_finished_SAW_sample1():
    saw = SAW(10)
    attempts = 1
    while True:
        px = 1
        saw.reset()
        while saw.getNumValidMoves() > 0:
            px *= saw.getNumValidMoves()
            saw.randomMove()
            if saw.pos == point(10,10):
                return SAWSample(saw.length,px/attempts, saw.history)
        attempts += 1
    return None

#method 2 of computing px


def draw_SAW_sample2():
    pass


#experiment set up
m = 10
batch = 3
draw_samples_fn = draw_SAW_samples1
path_fname = "test.h5"
fin_path_fname = "fin_test.h5"

max_length = 0
max_hist = None
fin_max_length = 0
fin_max_hist = None

with h5py.File(path_fname, 'w') as f,h5py.File(fin_path_fname,'w') as fin_f:
    dset = f.create_dataset('paths_info',(0,2),maxshape=(None,2),dtype='f',chunks=(batch,2))
    fin_dset = fin_f.create_dataset('fin_paths_info',(0,2),maxshape=(None,2),dtype='f',chunks=(batch,2))

    while m > 0:
        samples,fin_samples = draw_samples_fn(batch=batch)
        n_samples = len(samples)
        data = np.array([(sample.px, sample.length) for sample in samples]).astype(float)

        batch_max_idx = np.argmax(data[:,1])
        if max_length < data[batch_max_idx,1]:
            max_length = data[batch_max_idx,1]
            max_hist = samples[batch_max_idx].history
        dset.resize(dset.shape[0]+n_samples,axis=0) 
        dset[-n_samples:] = data
        
        data = np.array([(sample.px, sample.length) for sample in fin_samples]).astype(float)
        if len(data) > 0:
            batch_max_idx = np.argmax(data[:,1])
            if fin_max_length < data[batch_max_idx,1]:
                fin_max_length = data[batch_max_idx,1]
                fin_max_hist = samples[batch_max_idx].history
            fin_dset.resize(dset.shape[0]+n_samples,axis=0) 
            fin_dset[-n_samples:] = data
            m -= n_samples
