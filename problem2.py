from collections import namedtuple
from enum import Enum
import numpy as np
import multiprocessing as mp
from SAW import SAW
import csv
import pickle
import argparse

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
#method 2 of computing px (epsilon)
def draw_SAW_samples2(batch):
    saw = SAW(10)
    epsilon = 1e-8
    attempts = 0
    samples = []
    fin_samples = []
    for b in range(batch):
        saw.reset()
        px = 1
        attempts += 1
        while saw.getNumValidMoves() > 0:
            if np.random.random() < epsilon:
                break
            px *= saw.getNumValidMoves()
            saw.randomMove()
            if saw.pos == point(10,10):
                fin_samples.append(SAWSample(px/attempts, saw.length,saw.history))
                attempts = 0
        samples.append(SAWSample(px,saw.length, saw.history))
    return samples, fin_samples
def draw_SAW_samples3(batch):
    pass

draw_samples_fns = [draw_SAW_samples1,draw_SAW_samples2, draw_SAW_samples3]

#argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--samples', help='number of samples', type=int)
parser.add_argument('--batch', help='number of samples', type=int)
parser.add_argument('--fn', help='number of samples', type=int)
#TODO parameter checking
args = parser.parse_args()

#experiment set up
m = args.samples
batch = args.batch
fn_choice = args.fn
draw_samples_fn = draw_samples_fns[fn_choice] 
path_fname = "problem2/{}.csv".format(fn_choice)
fin_path_fname = "problem2/fin_{}.csv".format(fn_choice)

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

#save max history with pickle
with open('problem2/max_length_{}.pkl'.format(fn_choice),'wb') as f:
    pickle.dump(max_hist,f)
with open('problem2/fin_max_length_{}.pkl'.format(fn_choice),'wb') as f:
    pickle.dump(fin_max_hist,f)
