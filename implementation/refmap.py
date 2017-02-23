import re
import copy
from util import polar2cartesian, polar2origincartesian, graph_results
import numpy as np
import matplotlib.pyplot as plt


class RefMap:
    def __init__(self, filename=None, samplesize=-1, tolerance=0.1):
        self.points = []
        try:
            if filename is not None:
                with open(filename, 'r') as f:
                    l = f.readline()
                    while l:
                        point = l.rstrip('\n').split(",")
                        self.points.append(map(float, point))
                        l = f.readline()
                    # Data is in polar form 
        except ValueError as e:
            print "Error in file: ", filename
            raise e
        self.points = subsample(self.points, tolerance)

def subsample(points, tolerance):
    newpoints = []
    for p1 in points:
        if not containsSimilar(newpoints, p1, tolerance):
            newpoints.append(p1)
    return newpoints

def containsSimilar(set, p, tolerance):
    for p2 in set:
        if abs(p2[0]-p[0])<=tolerance and abs(p2[1]-p[1])<tolerance:
            return True
    return False

if __name__=="__main__":
    refmap = RefMap("data/combined.csv", tolerance=0.015)
    graph_results(refmap, [], (0,0,0))