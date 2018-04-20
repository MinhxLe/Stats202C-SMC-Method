from collections import namedtuple
from enum import Enum
import numpy as np
import multiprocessing as mp
from SAW import SAW
import csv
import pickle
import argparse
import copy 
point = namedtuple('point', ['x','y'])


SAWSample = namedtuple('SAWSample', \
        ['px','length','history'])

#method 1 of computing px
def draw_SAW_samples0(batch):
    saw = SAW(10)
    attempts = 0
    samples = []
    fin_samples = []
    for b in range(batch):
        px = 1
        saw.reset()
        attempts += 1
        while saw.getNumValidMoves() > 0:
            px *= saw.getNumValidMoves()
            saw.randomMove()
        if saw.pos == point(10,10):
            fin_samples.append(SAWSample(px/attempts, saw.length,copy.copy(saw.history)))
            attempts = 0
        samples.append(SAWSample(px,saw.length, copy.copy(saw.history)))
    return samples, fin_samples
#method 2 of computing px (epsilon)
def draw_SAW_samples1(batch):
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
            px *= saw.getNumValidMoves()/(1-epsilon)
            saw.randomMove()
        if saw.pos == point(10,10):
            fin_samples.append(SAWSample(np.prod(px)/attempts, saw.length,copy.copy(saw.history)))
            attempts = 0
        samples.append(SAWSample(np.prod(px),saw.length, copy.copy(saw.history)))
    return samples, fin_samples

#method 3 of computing px (epsilon)
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
        epsilon=1e-8
        while saw.getNumValidMoves() > 0:
            if np.random.random() < epsilon:
                break
            epsilon *= .5
            px *= saw.getNumValidMoves()/(1-epsilon)
            saw.randomMove()
        if saw.pos == point(10,10):
            fin_samples.append(SAWSample(np.prod(px)/attempts, saw.length,copy.copy(saw.history)))
            attempts = 0
        samples.append(SAWSample(np.prod(px),saw.length, copy.copy(saw.history)))
    return samples, fin_samples
def continue_SAW_sample3(saw,px,attempts,split):
    saw = copy.deepcopy(saw) 
    num_moves = saw.getNumValidMoves()
    fin_sample = None
    while num_moves > 0:
        px *= num_moves
        saw.randomMove()
        if saw.pos == point(10,10):
            fin_sample = SAWSample(px/split,saw.length,saw.history)
        num_moves = saw.getNumValidMoves()
    sample = SAWSample(px/split,saw.length,saw.history)
    return sample,fin_sample
def draw_SAW_samples3(batch):
    saw = SAW(10)
    attempts = 0
    samples = []
    fin_samples = []
    do_branch = False
    n_split = 5
    while batch > len(samples):
        saw.reset()
        px = 1
        attempts += 1
        while saw.getNumValidMoves() > 0:
            px *= saw.getNumValidMoves()
            saw.randomMove()
            if saw.pos == point(10,10):
                fin_samples.append(SAWSample(px/attempts, saw.length,saw.history))
                attempts = 0
            if saw.length >= 50:
                do_branch = True
                for i in range(0,n_split):
                    sample,fin_sample = continue_SAW_sample3(saw,px,attempts,n_split)
                    did_finish = False
                    if fin_sample:
                        fin_samples.append(fin_sample)
                        did_finish = True
                    else:
                        attempts += 1
                    if did_finish:
                        attempts = 0
                    samples.append(sample)
                break
        if not do_branch:
            samples.append(SAWSample(px,saw.length, saw.history))
        do_branch = False
    return samples, fin_samples

draw_samples_fns = [draw_SAW_samples0,draw_SAW_samples1, draw_SAW_samples2]

#argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--samples', help='number of samples', type=int)
parser.add_argument('--batch', help='number of samples', type=int)
parser.add_argument('--fn', help='number of samples', type=int)
parser.add_argument('--seed',help='random see',type=int)
#TODO parameter checking

#seed
args = parser.parse_args()
np.random.seed(args.seed)

#experiment set up
m = args.samples
batch = args.batch
fn_choice = args.fn
draw_samples_fn = draw_samples_fns[fn_choice] 
path_fname = "problem2/fn{}/{}.csv".format(fn_choice,args.seed)
fin_path_fname = "problem2/fin_fn{}/{}.csv".format(fn_choice,args.seed)

max_length = 0
max_hist = None
fin_max_length = 0
fin_max_hist = None

with open(path_fname, 'w+') as f,open(fin_path_fname,'w+') as fin_f:
    writer = csv.writer(f)
    fin_writer = csv.writer(fin_f)
    counter = m
    while counter > 0:
        print(counter)
        samples,fin_samples = draw_samples_fn(batch=batch)
        data = np.array([(sample.px, sample.length) for sample in samples]).astype(float)
        counter -= len(data)
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
            fin_writer.writerows(data)
    f.flush()
    fin_f.flush()
#save max history with pickle
if max_hist:
    with open('problem2/fn{}/max_length{}.pkl'.format(fn_choice,args.seed),'wb') as f:
        pickle.dump(max_hist,f)
if fin_max_hist:
    with open('problem2/fin_fn{}/max_length{}.pkl'.format(fn_choice,args.seed),'wb') as f:
        pickle.dump(fin_max_hist,f)
