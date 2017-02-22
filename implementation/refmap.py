import re
import copy
from util import polar2cartesian, polar2origincartesian
import numpy as np
import matplotlib.pyplot as plt


class RefMap:

    def __init__(self, filename=None, samplesize=-1):
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
        # self.points = [p for p in self.points if -10<p[0]<2 and -10<p[1]<2]
        if samplesize!=-1:
            ratio = len(self.points)/samplesize
            self.points = self.points[::ratio]






