from scanreader import Scan
import matplotlib.pyplot as plt
from util import *
from refmap import RefMap
# Draws GA pose estimation from best individual
scanName = "scans/scan110"
tolerance = 0.2
individual = (-2.184327,2.641909,1.1352500)

refmap = RefMap("data/combined.csv", tolerance=tolerance).points
errorscan = Scan(scanName, tolerance=tolerance)
graph_results(refmap, errorscan.scan_points, individual)