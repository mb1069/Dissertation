import sys

sys.path.append('../src/util')

from refmap import RefMap
from scanreader import Scan
from util import graph_results
import matplotlib.pyplot as plt
# Draws GA pose estimation from best individual


if __name__=="__main__":
	scanName = "../scans/scan110"
	tolerance = 0.2
	individual = (-3.0340085645486083, 1.4682359473584328, 0.725126269477046)

	refmap = RefMap("../data/combined.csv", tolerance=tolerance).points
	errorscan = Scan(scanName, tolerance=tolerance)
	graph_results(refmap, errorscan.scan_points, individual)