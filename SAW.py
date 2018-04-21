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
        self.history = []
    def randomMove(self):
        p = self.pos
        self.history.append(p)
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
            self.length += 1
    
    def getNumValidMoves(self):
        p = self.pos      
        return self.isValidPos(point(p.x-1,p.y)) \
                + self.isValidPos(point(p.x+1,p.y)) \
                + self.isValidPos(point(p.x,p.y-1)) \
                + self.isValidPos(point(p.x,p.y+1))
    def generateDisplay(self):
        pass

