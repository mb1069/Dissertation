from scanreader import Scan
import matplotlib.pyplot as plt
from util import *
from refmap import RefMap
# Draws GA pose estimation from best individual
scanName = "scans/scan110"
tolerance = 0.2
individual = (-2.45697176655,2.85781308878,1.03606367248,)

refmap = RefMap("data/combined.csv", tolerance=tolerance).points
errorscan = Scan(scanName, tolerance=tolerance)
graph_results(refmap, errorscan.scan_points, individual)