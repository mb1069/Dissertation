import re
import copy
import math

reg = re.compile('(?:.+[:])?([-|0-9|.]+)+')

def polar2cartesian(dist, ang):
    return [dist * math.cos(ang), dist * math.sin(ang)]

def applyRotation(x, y, rotErr):
    return (x*math.cos(rotErr)) - y*(math.sin(rotErr)), (x*math.sin(rotErr)) + y*(math.cos(rotErr))

def applyTranslation(x, y, xErr, yErr):
    return x+xErr, y+yErr

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
            # Handle inconsistent file formatting
            if len(scan_loc) > 3:
                self.scan_points = polar2cartesian(scan_loc[3], scan_loc[4])
            line = f.readline()
            # Data is in polar form 
            while line:
                coords = map(float, reg.findall(line))
                self.scan_points.append(polar2cartesian(coords[0], coords[1]))
                line = f.readline()

    def applyError(self, xErr, yErr, rotErr):
        data = []
        for i in range(0, len(self.scan_points)-1):
            x, y = self.scan_points[i]
            # Applying rotation
            print x, y
            x, y = applyRotation(x, y, rotErr)
            x, y = applyTranslation(x, y, xErr, yErr)
            print x,y
            print 
            data.append([x, y])
        return data


scan = Scan("data/scan0")
scan2 = scan.applyError(0, 0, 1)

for x in scan.scan_points[0:10]:
    print x[0], x[1], "\r"
print
print

for x in scan2[0:10]:
    print x[0], x[1], "\r"