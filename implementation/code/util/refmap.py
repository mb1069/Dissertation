import re
import copy
from util import polar2cartesian, polar2origincartesian, graph_results, subsample
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





if __name__=="__main__":
    refmap = RefMap("data/combined.csv", tolerance=0.015)
    graph_results(refmap, [], (0,0,0))