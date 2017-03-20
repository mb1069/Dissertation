import re
import copy
from util import polar2cartesian, polar2origincartesian, graph_results, subsample
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from tqdm import trange
class LookupRefMap:
    def __init__(self,  filename, numpoints, samplesize=-1, tolerance=0.1):
        self.points = []
        self.minX = sys.maxint
        self.minY = sys.maxint
        self.maxX = -sys.maxint - 1
        self.maxY = -sys.maxint - 1
        try:
            if filename is not None:
                with open(filename, 'r') as f:
                    l = f.readline()
                    while l:
                        point = map(float, l.rstrip('\n').split(","))
                        if point[0]<self.minX:
                            self.minX = point[0]
                        elif point[0]>self.maxX:
                            self.maxX = point[0]
                        if point[1]<self.minY:
                            self.minY = point[1]
                        elif point[1]>self.maxY:
                            self.maxY = point[1]
                        self.points.append(point)
                        l = f.readline()
                    # Data is in polar form 
        except ValueError as e:
            print "Error in file: ", filename
            raise e

        self.points = subsample(self.points, tolerance)

        dimX = self.maxX-self.minX
        dimY = self.maxY-self.minY
        Xpoints = int(math.sqrt(((dimX*numpoints)/dimY)+(math.pow(dimX-dimY, 2)/(4*(dimY**2))))-((dimX-dimY)/(2*dimY)))
        Ypoints = int(numpoints/(Xpoints))
        self.Xstep = dimX/(Xpoints-1)
        self.Ystep = dimY/(Ypoints-1)
        
        self.grid = []
        for x in trange(Xpoints):
            row = []
            for y in range(Ypoints):
                Xrange = (x*self.Xstep, (x+1)*self.Xstep)
                Yrange = (y*self.Xstep, (y+1)*self.Ystep)
                hasPoint = False
                for point in self.points:
                    if inRange(point[0], Xrange) & inRange(point[1], Yrange):
                        hasPoint = True
                        break
                row.append(1 if hasPoint else 0)
            self.grid.append(row)

    def evaluatePoint(self, point):
        Xcell = int(point[0]/self.Xstep)
        Ycell = int(point[1]/self.Ystep)
        return self.grid[Xcell][Ycell]


def inRange(value, rangetup):
    return rangetup[0]<=value<=rangetup[1]

