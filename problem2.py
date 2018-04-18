from collections import namedtuple
from enum import Enum
import numpy as np
import multiprocessing as mp
from SAW import SAW
import csv

point = namedtuple('point', ['x','y'])


SAWSample = namedtuple('SAWSample', \
        ['px','length','history'])

#method 1 of computing px
def draw_SAW_samples1(batch):
    saw = SAW(10)
    attempts = 0
    samples = []
    fin_samples = []
    for b in range(batch):
        saw.reset()
        px = 1
        attempts += 1
        while saw.getNumValidMoves() > 0:
            px *= saw.getNumValidMoves()
            saw.randomMove()
            if saw.pos == point(10,10):
                fin_samples.append(SAWSample(px/attempts, saw.length,saw.history))
                attempts = 0
        samples.append(SAWSample(px,saw.length, saw.history))
    return samples, fin_samples
#method 2 of computing px
def draw_SAW_sample2():
    saw = SAW(10)
    epsilon = 1e-6
    attempts = 0
    samples = []
    fin_samples = []
    for b in range(batch):
        saw.reset()
        px = 1
        attempts += 1
        while saw.getNumValidMoves() > 0:
            px *= saw.getNumValidMoves()
            saw.randomMove()
            if saw.pos == point(10,10):
                fin_samples.append(SAWSample(px/attempts, saw.length,saw.history))
                attempts = 0
            if np.random.random() > epsilon:
                break
        samples.append(SAWSample(px,saw.length, saw.history))
    return samples, fin_samples

#experiment set up
m = 1000000
batch =10000 
draw_samples_fn = draw_SAW_samples1
path_fname = "problem2/2.csv"
fin_path_fname = "problem2/fin_2.csv"

max_length = 0
max_hist = None
fin_max_length = 0
fin_max_hist = None

n_fin_samples = 0
with open(path_fname, 'w+') as f,open(fin_path_fname,'w+') as fin_f:
    writer = csv.writer(f)
    fin_writer = csv.writer(fin_f)

    while m > 0:
        samples,fin_samples = draw_samples_fn(batch=batch)
        n_samples = len(samples)
        data = np.array([(sample.px, sample.length) for sample in samples]).astype(float)

        batch_max_idx = np.argmax(data[:,1])
        if max_length < data[batch_max_idx,1]:
            max_length = data[batch_max_idx,1]
            max_hist = samples[batch_max_idx].history
        writer.writerows(data)

        data = np.array([(sample.px, sample.length) for sample in fin_samples]).astype(float)
        if len(data) > 0:
            batch_max_idx = np.argmax(data[:,1])
            if fin_max_length < data[batch_max_idx,1]:
                fin_max_length = data[batch_max_idx,1]
                fin_max_hist = samples[batch_max_idx].history
            m -= n_samples
        fin_writer.writerows(data)
