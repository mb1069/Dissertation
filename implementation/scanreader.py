import re

reg = re.compile('(?:.+[:])?([-|0-9|.]+)+')


class Scan:
    posx = 0.0
    posy = 0.0
    rot = 0.0
    scan_points = []

    def __init__(self, filename):
        f = open(filename, 'r')
        # Get the scan's pose
        scan_loc = reg.findall(f.readline())
        self.posx = float(scan_loc[0])
        self.posy = float(scan_loc[1])
        self.rot = float(scan_loc[2])
        # Handle inconsistent file formatting
        if len(scan_loc) > 3:
            self.scan_points = [scan_loc[3], scan_loc[4]]
        line = f.readline()
        while line:
            coords = map(float, reg.findall(line))
            self.scan_points.append(coords)
            line = f.readline()


a = Scan("data/scan0")
