import multiprocessing
import numpy as np
import random
import math
import argparse as ap
import copy
from tqdm import trange, tqdm
from deap import base
from deap import creator
from deap import tools

from util.deap_custom import eaSimple
from util.refmap import RefMap
from util.scanreader import Scan
from util.util import hausdorff, applytuple, graph_results, total_sum, save_data, evaluate_solution


NGEN = 200
POP = 200
CXPB = 0.15
MUTPB = 0.05

MIN = -10
MAX = 10

TRANS_MIN, TRANS_MAX = -8.0, 8.0
ROT_MIN, ROT_MAX = -math.pi, math.pi

scanName = None
refmap = None
errorscan = None

scanName = "scans/scan110"


# errorscan.scan_points.append((-2.184327,2.641909))
# graph_results(refmap, errorscan.scan_points, (0,0,0))


def evaluate(individual):
    dataset = applytuple(errorscan.scan_points, *individual)
    return 1/(1+total_sum(dataset, refmap)),
    # return 1/(1+hausdorff(dataset, refmap)),

def main(multicore, NGEN, POP, scan, map, CXPB, MUTPB, verb):



    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_trans", random.uniform, TRANS_MIN, TRANS_MAX)
    toolbox.register("attr_rot", random.uniform, ROT_MIN, ROT_MAX)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_trans, toolbox.attr_trans, toolbox.attr_rot), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, sigma=0.125/4, mu=0, indpb=MUTPB)
    # toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=MUTPB)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("evaluate", evaluate)
    if multicore:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    

    random.seed()
    record, log = eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                   stats=stats, halloffame=hof, verbose=args.v)
    expr = tools.selBest(pop, 1)[0]
    if verb:
        print "Best individual:", expr
        print "Fitness:", evaluate_solution(expr[0], expr[1], expr[2], errorscan.posx, errorscan.posy, errorscan.rot)
    return evaluate(expr)[0], record, log, expr

if __name__ == "__main__":

    parser = ap.ArgumentParser(description="My Script")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--multicore", action='store_true')
    # parser.add_argument("--seed")
    parser.add_argument("--save", type=str, default="temp.csv")
    parser.add_argument("-v", action='store_true', default=False)
    parser.add_argument("--graph", action='store_true', default=False)
    # parser.add_argument("--max_gen", type=int)
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gen", type=int, default=200)

    args, leftovers = parser.parse_known_args()
    # Using full map
    refmap = RefMap("../data/combined.csv", tolerance=args.tolerance).points

    # Using error
    # refmap = Scan(scanName)
    # refmap = applytuple(refmap.scan_points, refmap.posx, refmap.posy, refmap.rot)
    errorscan = Scan("../"+scanName, tolerance=args.tolerance)

    print "Aiming for"
    print errorscan.posx, errorscan.posy, errorscan.rot
    for NGEN in tqdm(np.arange(450, 500, 50)):
        for x in trange(args.iterations):
            best_fitness, record, log, expr = main(multicore = args.multicore, verb=args.v, POP = args.pop, NGEN = args.gen, scan=copy.deepcopy(errorscan), map=refmap, CXPB=CXPB, MUTPB=MUTPB)
            if args.save is not None:
                row = [NGEN, best_fitness, expr[0], expr[1], expr[2], "\r"]
                save_data(row, "../results/"+args.save)
