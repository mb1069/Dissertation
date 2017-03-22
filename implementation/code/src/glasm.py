import multiprocessing
import numpy as np
import random
import os.path
import math
import argparse as ap
import copy
import pickle

from tqdm import trange, tqdm
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from util.lookupmap import LookupRefMap
from util.scanreader import Scan
# from util.deap_custom import eaSimple
from util.util import hausdorff, applytuple, graph_results, total_sum, save_data, evaluate_solution, graph_gen, update_series, initPop, lookup_total_sum, pickledMapExists
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
target = None

scanName = "../scans/scan110"

pickleFolder = "../glasm_maps/"

# errorscan.scan_points.append((-2.184327,2.641909))
# graph_results(refmap, errorscan.scan_points, (0,0,0))


def evaluate(individual):
    dataset = applytuple(errorscan.scan_points, *individual)
    return lookup_total_sum(refmap, dataset),
    # return 1/(1+hausdorff(dataset, refmap)),

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, graph=False):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    if graph:
        graph, pop_series = graph_gen(refmap.points, population, target)

    # Begin the generational process

    # Hide TQDM pbar if verbose, as logbook will be printed
    pbar = range(0,ngen) if verbose else trange(ngen, leave=False)
    for gen in pbar:
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        desc = str(toolbox.evaluate(tools.selBest(population, 1)[0])[0])
        if graph:
            update_series(graph, pop_series, population)

        if verbose:
            print tools.selBest(population, 1)[0]

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream
        else:
            pbar.set_description(desc)

    return record, logbook


def main(multicore, NGEN, POP, scan, map, CXPB, MUTPB, verb, graph):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_trans", random.uniform, TRANS_MIN, TRANS_MAX)
    toolbox.register("attr_rot", random.uniform, ROT_MIN, ROT_MAX)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_trans, toolbox.attr_trans, toolbox.attr_rot), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, sigma=0.125/4, mu=0, indpb=MUTPB)
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
                                   stats=stats, halloffame=hof, verbose=args.v, graph=args.graph)
    expr = tools.selBest(pop, 1)[0]
    if verb:
        print "Best individual:", expr
        print "Fitness:", evaluate_solution(expr[0], expr[1], expr[2], errorscan.posx, errorscan.posy, errorscan.rot)
    best_result = evaluate_solution(expr[0], expr[1], expr[2], errorscan.posx, errorscan.posy, errorscan.rot)
    return best_result, record, log, expr



if __name__ == "__main__":
    parser = ap.ArgumentParser(description="My Script") 
    parser.add_argument("--iterations", type=int, default=1) 
    parser.add_argument("--multicore", action='store_true', default=False) 
    # parser.add_argument("--seed") 
    parser.add_argument("--savefile", type=str, default="temp.csv") 
    parser.add_argument("-v", action='store_true', default=False) 
    parser.add_argument("--graph", action='store_true', default=False) 
    # parser.add_argument("--max_gen", type=int) 
    parser.add_argument("--tolerance", type=float, default=0.2) 
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gen", type=int, default=50)
    parser.add_argument("--numcells", type=int, default=100000)

    args, leftovers = parser.parse_known_args()
    pickle_file_name = pickleFolder+"tol="+str(args.tolerance)+"cells="+str(args.numcells)

    if pickledMapExists(pickle_file_name):
        print "Found and loading pickled map"
        refmap = pickle.load(open(pickle_file_name, "rb"))
    else:
        print "Map not found, pickling map"
        refmap = LookupRefMap("../data/combined.csv", args.numcells, tolerance=args.tolerance)
        pickle.dump(refmap, open(pickle_file_name, "wb"))
        print "Loaded and pickled map for further use"

    save_data([__file__, "pop:"+str(args.pop), "gen:"+str(args.gen), "grid:"+str(args.grid), "numcells:"+str(args.numcells), "\r"], "../results/"+args.savefile)

    # Using full map

    errorscan = Scan(scanName, tolerance=args.tolerance)
    target = (errorscan.posx, errorscan.posy, errorscan.rot)
    for x in trange(args.iterations):
        best_fitness, record, log, expr = main(multicore = args.multicore, verb=args.v, POP = args.pop, NGEN = args.gen, scan=copy.deepcopy(errorscan), map=refmap, CXPB=CXPB, MUTPB=MUTPB, graph=args.graph)
        if args.savefile is not None:
            row = [best_fitness, expr[0], expr[1], expr[2], "\r"]
            save_data(row, "../results/"+args.savefile)
