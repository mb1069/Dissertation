import math
from ga import evaluate
import argparse as ap
from refmap import RefMap
from scanreader import Scan
import numpy as np
from tqdm import trange, tqdm
from util import applytuple, total_sum, save_data
import multiprocessing as mp
import os

"""
File to brute force a fitness landscape over a map
"""
errorscan = None
refmap = None
filename = None
def evaluate(individual, errorscan, refmap):
    dataset = applytuple(errorscan.scan_points, *individual)
    return 1/(1+total_sum(dataset, refmap)),

def evaluate_pose(ind):
	fitness = evaluate(ind, errorscan, refmap)[0]
	row = [ind[0], ind[1], ind[2], str(fitness),"\r"]
	save_data(row, "results/"+filename)

def main(scan_name):
	global errorscan
	global refmap
	global filename

	parser = ap.ArgumentParser(description="My Script")
	parser.add_argument("--savefile", type=str, default="landscape.csv")
	parser.add_argument("-v", action='store_true', default=False)
	parser.add_argument("--graph", action='store_true', default=False)
	parser.add_argument("--tolerance", type=float, default=0.2)

	args, leftovers = parser.parse_known_args()

	scanName = "scans/"+scan_name
	refmap = RefMap("data/combined.csv", tolerance=args.tolerance).points
	filename = "landscape_multirot.csv"


	errorscan = Scan(scanName, tolerance=args.tolerance)
	xs = map(float, np.arange(-8, 8, 0.25))
	ys = map(float, np.arange(-8, 8, 0.25))
	rots = map(float, np.arange(-math.pi, math.pi, math.pi/12))
	total = len(xs) * len(ys) * len(rots)
	step = len(rots)
	inds = []
	for x in xs:
		for y in ys:
			for rot in rots:
				inds.append([x,y,rot])
	pool = mp.Pool(mp.cpu_count())
	print "starting"
	for x in tqdm(pool.imap_unordered(evaluate_pose, inds), total=len(inds)):
		pass

if __name__=="__main__":
	for x in trange(0, 500, 25):
		scanName = "scan"+str(x)
		main(scanName)
