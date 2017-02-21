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
                f = open(filename, 'r')
                l = f.readline()
                while l:
                    point = l.rstrip('\n').split(",")
                    self.points.append(map(float, point))
                    l = f.readline()
                # Data is in polar form 
                f.close()
        except ValueError as e:
            print "Error in file: ", filename
            raise e
        if samplesize!=-1:
            ratio = len(self.points)/samplesize
            self.points = self.points[::ratio]






