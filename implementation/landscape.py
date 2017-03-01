import math
from ga import evaluate
import argparse as ap
from refmap import RefMap
from scanreader import Scan
import numpy as np
from tqdm import tqdm
from util import applytuple, total_sum, save_data
import multiprocessing as mp


"""
File to brute force a fitness landscape over a map
"""

def evaluate(individual, errorscan, refmap):
    dataset = applytuple(errorscan.scan_points, *individual)
    return 1/(1+total_sum(dataset, refmap)),






def evaluate_pose(ind):
	fitness = evaluate(ind, errorscan, refmap)[0]
	row = [ind[0], ind[1], ind[2], str(fitness),"\r"]
	save_data(row, "results/"+filename)

if __name__=="__main__":
	parser = ap.ArgumentParser(description="My Script")
	parser.add_argument("scan_name", type=str)
	parser.add_argument("--savefile", type=str, default="landscape.csv")
	parser.add_argument("-v", action='store_true', default=False)
	parser.add_argument("--graph", action='store_true', default=False)
	parser.add_argument("--tolerance", type=float, default=0.2)

	args, leftovers = parser.parse_known_args()

	scanName = "scans/"+args.scan_name
	refmap = RefMap("data/combined.csv", tolerance=args.tolerance).points
	filename = "landscape_multirot.csv"


	errorscan = Scan(scanName, tolerance=args.tolerance)
	xs = map(float, np.arange(-8, 8, 0.25))
	ys = map(float, np.arange(-8, 8, 0.25))
	rots = map(float, np.arange(-math.pi, math.pi, math.pi/12))
	total = len(xs) * len(ys) * len(rots)
	pbar = tqdm(total=total)
	step = len(rots)
	inds = []
	for x in xs:
		for y in ys:
			for rot in rots:
				inds.append([x,y,rot])
			pbar.update(step)
	pbar.close()

	pool = mp.Pool(mp.cpu_count())
	pool.map(evaluate_pose, inds)

