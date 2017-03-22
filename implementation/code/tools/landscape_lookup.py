import math
import multiprocessing as mp
import os, sys
from tqdm import trange, tqdm
import numpy as np
import argparse as ap
import pickle

sys.path.append('../src/util')

# from src.ga import evaluate
from util.lookupmap import LookupRefMap
from util.scanreader import Scan
from util.util import applytuple, total_sum, save_data, lookup_total_sum, pickledMapExists


"""
File to brute force a fitness landscape over a map
"""
errorscan = None
refmap = None
filename = None
pickleFolder = "../glasm_maps/"

def evaluate(individual, errorscan, refmap):
    dataset = applytuple(errorscan.scan_points, *individual)
    return lookup_total_sum(refmap, dataset),

def evaluate_pose(ind):
	fitness = evaluate(ind, errorscan, refmap)[0]
	row = [ind[0], ind[1], ind[2], str(fitness),"\r"]
	save_data(row, "../results/"+filename)

def main():
	global errorscan
	global refmap
	global filename

	parser = ap.ArgumentParser(description="My Script")
	parser.add_argument("--savefilefile", type=str, default="landscape.csv")
	parser.add_argument("--scan", type=str, default="scan110")
	parser.add_argument("-v", action='store_true', default=False)
	parser.add_argument("--graph", action='store_true', default=False)
	parser.add_argument("--tolerance", type=float, default=0.2)
	parser.add_argument("--numcells", type=int, default=1000000)
	args, leftovers = parser.parse_known_args()

	scanName = "../scans/"+args.scan
	
	pickle_file_name = pickleFolder+"tol="+str(args.tolerance)+"cells="+str(args.numcells)
	if pickledMapExists(pickle_file_name):
		print "Found and loading pickled map"
		refmap = pickle.load(open(pickle_file_name, "rb"))
	else:
		print "Map not found, pickling map"
		refmap = LookupRefMap("../data/combined.csv", args.numcells, tolerance=args.tolerance)
		pickle.dump(refmap, open(pickle_file_name, "wb"))
		print "Loaded and pickled map for further use"

	filename = args.savefilefile


	errorscan = Scan(scanName, tolerance=args.tolerance)
	xs = map(float, np.arange(-8, 8, 0.125))
	ys = map(float, np.arange(-8, 8, 0.125))
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
	main()
