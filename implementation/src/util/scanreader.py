import re
import copy
from util import polar2cartesian, polar2origincartesian, subsample
import numpy as np
import matplotlib.pyplot as plt

reg = re.compile(
    "(?:^scan\.pos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)(-?\d\.\d+),(-?\d+.\d+)$)|(?:^(-?\d+\.\d+),(-?\d+\.\d+)$)")


class Scan:

    # Retrieve scan with cartesian coordinates relative to pose
    def __init__(self, filename=None, absolute=False, tolerance=0.1):
        try:
            if filename is not None:
                f = open(filename, 'r')
                # Get the scan's pose (stored in cartesian form)
                l = f.readline()
                scan_loc = reg.findall(l)[0]
                self.posx = float(scan_loc[0])
                self.posy = float(scan_loc[1])
                self.rot = float(scan_loc[2])
                self.scan_points = []
                if absolute:
                    self.scan_points.append(polar2origincartesian(self, scan_loc[3], scan_loc[4]))
                else:
                    self.scan_points.append(polar2cartesian(scan_loc[3], scan_loc[4]))

                line = f.readline()
                # Data is in polar form 
                while line:
                    coords = map(float, reg.findall(line)[0][5:7]) 
                    if absolute:
                        self.scan_points.append(polar2origincartesian(self, coords[0], coords[1]))
                    else:
                        self.scan_points.append(polar2cartesian(coords[0], coords[1]))

                    line = f.readline()
                f.close()
        except ValueError as e:
            print "Error in file: ", filename
            raise e
        self.scan_points = subsample(self.scan_points, tolerance)

    def getcopy(self):
        return copy.deepcopy(self)

# scan = Scan("scans/scan0")
# scan = Scan("scans/scan5")
# print polar2cartesian(4, 0.5)
# scan2 = scan.applytuple(0, 0, 1)
#
# for x in scan.scan_points[0:10]:
#     print x[0], x[1], "\r"
# print
# print
#
# for x in scan2[0:10]:
#     print x[0], x[1], "\r"
