import re
import copy
from util import polar2cartesian
reg = re.compile("(?:.+[:])?([-|0-9|.]+)+")


class Scan:
    posx = None
    posy = None
    rot = None
    scan_points = []

    def __init__(self, filename=None):
        if filename is not None:
            f = open(filename, 'r')
            # Get the scan's pose (stored in cartesian form)
            scan_loc = reg.findall(f.readline())
            self.posx = float(scan_loc[0])
            self.posy = float(scan_loc[1])
            self.rot = float(scan_loc[2])
            self.scan_points = []
            # Handle inconsistent file formatting
            if len(scan_loc) > 3:
                self.scan_points.append(polar2cartesian(scan_loc[3], scan_loc[4]))
            line = f.readline()
            # Data is in polar form 
            while line:
                coords = map(float, reg.findall(line))
                self.scan_points.append(polar2cartesian(coords[0], coords[1]))
                line = f.readline()



    def getcopy(self):
        return copy.deepcopy(self)




# scan = Scan("data/scan0")
# scan2 = scan.applytuple(0, 0, 1)
#
# for x in scan.scan_points[0:10]:
#     print x[0], x[1], "\r"
# print
# print
#
# for x in scan2[0:10]:
#     print x[0], x[1], "\r"
