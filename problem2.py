from collections import namedtuple
from enum import Enum
import numpy as np
import multiprocessing as mp
import csv
point = namedtuple('point', ['x','y'])
class SAW:
    def __init__(self, n):
        self.pos = point(0,0)
        self.grid = np.zeros([n+1,n+1],dtype=bool)
        self.history =[point(0,0)]
        self.length = 0
        self.n = n
    def isValidPos(self,pos):
        return 0 <= pos.x and pos.x <= self.n \
                and 0 <= pos.y and pos.y <= self.n \
                and not self.grid[pos.x,pos.y]
    def reset(self):
        self.pos = point(0,0)
        self.grid = np.zeros([self.n+1,self.n+1],dtype=bool)
        self.length = 0
    def randomMove(self):
        p = self.pos
        self.grid[p.x,p.y] = True #set current pos as traveled
        directions = []
        #right
        if self.isValidPos(point(p.x+1,p.y)):
            directions.append(point(p.x+1,p.y))    
        #left
        if self.isValidPos(point(p.x-1,p.y)): 
            directions.append(point(p.x-1,p.y))    
        #up        
        if self.isValidPos(point(p.x,p.y-1)): 
            directions.append(point(p.x,p.y-1))    
        #down
        if self.isValidPos(point(p.x,p.y+1)):
            directions.append(point(p.x,p.y+1))
        if len(directions) > 0:
            self.pos = directions[np.random.randint(len(directions))]
            self.history.append(self.pos)
            self.length += 1
    
    def getNumValidMoves(self):
        p = self.pos      
        return self.isValidPos(point(p.x-1,p.y)) \
                + self.isValidPos(point(p.x+1,p.y)) \
                + self.isValidPos(point(p.x,p.y-1)) \
                + self.isValidPos(point(p.x,p.y+1))
    def generateDisplay(self):
        pass




SAWSample = namedtuple('SAWSample', \
        ['length','px','history'])
def draw_SAW_sample():
    saw = SAW(10)
    px = 1
    finishedSample = None
    while saw.getNumValidMoves() > 0:
        px *= saw.getNumValidMoves()
        saw.randomMove()
        if saw.pos == point(10,10):
            finishedSample = SAWSample(saw.length,px, saw.history)
    sample = SAWSample(saw.length,px, saw.history)
    return sample,finishedSample

# def mc(iter=int(100),fname):
    #pool = mp.Pool(4)
    #future_samples = [pool.apply_async(draw_SAW_sample) for _ in range(iter)]
    #samples = [f.get() for f in future_samples]
    #return samples

#experiment set up
m = int(1e8)

max_length = 0
max_hist = None
max_fin_length = 0
max_fin_hist = None

writer = csv.writer(open("problem2/1.csv",'w+'))
fin_writer = csv.writer(open("problem2/fin_1.csv",'w+'))
#TODO make this a batch instead
#TODO multiprocessing
for i in range(m):
    samples, finished_samples = draw_SAW_sample()
    length, px,history = samples
    if max_length < length:
        max_length = length
        max_hist = history
    writer.writerow((length,px))
    if finished_samples:
        length, px,history = finished_samples
        if max_length < length:
            max_fin_length = length
            max_fin_hist = history
        fin_writer.writerow((length,px))

