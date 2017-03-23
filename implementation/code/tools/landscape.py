import math
import multiprocessing as mp
import os, sys
from tqdm import trange, tqdm
import numpy as np
import argparse as ap

sys.path.append('../src/util')

# from src.ga import evaluate
from util.refmap import RefMap
from util.scanreader import Scan
from util.util import applytuple, total_sum, save_data


"""
File to brute force a fitness landscape over a map
"""
errorscan = None
refmap = None
filename = None
def evaluate(individual, errorscan, refmap):
    dataset = applytuple(errorscan.scan_points, *individual)
    return 1.0/(1.0+float(total_sum(dataset, refmap))),

def evaluate_pose(ind):
	fitness = evaluate(ind, errorscan, refmap)[0]
	row = [ind[0], ind[1], ind[2], str(fitness),"\r"]
	print row
	raw_input()
	save_data(row, "../results/"+filename)

def main():
	global errorscan
	global refmap
	global filename

	parser = ap.ArgumentParser(description="My Script")
	parser.add_argument("--savefile", type=str, default="landscape.csv")
	parser.add_argument("--scan", type=str, default="scan110")
	parser.add_argument("-v", action='store_true', default=False)
	parser.add_argument("--graph", action='store_true', default=False)
	parser.add_argument("--tolerance", type=float, default=0.2)

	args, leftovers = parser.parse_known_args()

	scanName = "../scans/"+args.scan
	refmap = RefMap("../data/combined.csv", tolerance=args.tolerance).points
	filename = args.savefile


	errorscan = Scan(scanName, tolerance=args.tolerance)
	xs = map(float, np.arange(-2.5, -2, 0.01))
	ys = map(float, np.arange(2.5, 3, 0.01))
	rots = map(float, np.arange(-math.pi, math.pi, math.pi/12))
	# -2.184327,2.641909,1.135250
	xs = [-2.184327]
	ys = [2.641909]
	rots = [1.135250]
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
	main()
