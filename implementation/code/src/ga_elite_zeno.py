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
from deap import algorithms

from util.refmap import RefMap
from util.scanreader import Scan
from util.deap_custom import varOrZeno
from util.util import hausdorff, applytuple, graph_results, total_sum, save_data, evaluate_solution, graph_gen, update_series, initPop


NGEN = 200
POP = 200
CXPB = 0.0
MUTPB = 1.0

MIN = -10
MAX = 10
elite = 0.5



TRANS_MIN, TRANS_MAX = -8.0, 8.0
ROT_MIN, ROT_MAX = -math.pi, math.pi

scanName = None
refmap = None
errorscan = None
target = None
scanName = "scans/scan110"


# errorscan.scan_points.append((-2.184327,2.641909))
# graph_results(refmap, errorscan.scan_points, (0,0,0))


def evaluate(individual):
    dataset = applytuple(errorscan.scan_points, *individual)
    return 1/(1+total_sum(dataset, refmap)),
    # return 1/(1+hausdorff(dataset, refmap)),

def eaMuPlusLambdaZeno(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, mu_mean, mu_sigma,
                   stats=None, halloffame=None, verbose=__debug__, graph=False):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream
    if graph:
        graph, pop_series = graph_gen(refmap, population, target)
    pbar = range(0,ngen) if verbose else trange(ngen, leave=False)
    for gen in pbar:
        # Vary the population
        progress = float(gen)/float(ngen)
        offspring = varOrZeno(population, toolbox, lambda_, cxpb, mutpb, 1-progress, mu_mean, mu_sigma)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if graph:
            update_series(graph, pop_series, population)
        if verbose:
            print logbook.stream
        else:
            desc = str(toolbox.evaluate(tools.selBest(population, 1)[0])[0])
            pbar.set_description(desc)

    return record, logbook

def main(multicore, NGEN, POP, refmap, MUTPB, mu_mean, mu_sigma, verb, grid, graph):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_trans", random.uniform, TRANS_MIN, TRANS_MAX)
    toolbox.register("attr_rot", random.uniform, ROT_MIN, ROT_MAX)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_trans, toolbox.attr_trans, toolbox.attr_rot), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if grid:
        pop = initPop(POP, refmap, creator.Individual)
    else:
        pop = toolbox.population(n=POP)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("evaluate", evaluate)

    if multicore:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    random.seed()
    record, log = eaMuPlusLambdaZeno(pop, toolbox, mu=int(POP*elite), lambda_=int(POP*(1-elite)), cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, mu_mean=mu_mean, mu_sigma=mu_sigma, stats=stats, halloffame=hof, verbose=args.v, graph=graph)
    expr = tools.selBest(pop, 1)[0]
    if verb:
        print "Best individual:", expr
        print "Fitness:", evaluate_solution(expr[0], expr[1], expr[2], errorscan.posx, errorscan.posy, errorscan.rot)
    best_result = evaluate_solution(expr[0], expr[1], expr[2], errorscan.posx, errorscan.posy, errorscan.rot)
    return best_result, record, log, expr

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="My Script")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--multicore", action='store_true')
    # parser.add_argument("--seed")
    parser.add_argument("--savefile", type=str, default="temp.csv")
    parser.add_argument("-v", action='store_true', default=False)
    parser.add_argument("--disp", action='store_true', default=False)
    # parser.add_argument("--max_gen", type=int)
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gen", type=int, default=50)
    parser.add_argument("--grid", action='store_true', default=False)
    parser.add_argument("--graph", action='store_true', default=False)

    args, leftovers = parser.parse_known_args()
    # Using full map
    refmap = RefMap("../data/combined.csv", tolerance=args.tolerance).points

    # Using error
    # refmap = Scan(scanName)
    # refmap = applytuple(refmap.scan_points, refmap.posx, refmap.posy, refmap.rot)
    errorscan = Scan("../"+scanName, tolerance=args.tolerance)
    target = (errorscan.posx, errorscan.posy, errorscan.rot)

    save_data([__file__, "pop:"+str(args.pop), "gen:"+str(args.gen), "grid:"+str(args.grid), "\r"], "../results/"+args.savefile)

    for x in trange(args.iterations):
        best_fitness, record, log, expr = main(multicore = args.multicore, verb=args.v, POP = args.pop, NGEN = args.gen, refmap=refmap, MUTPB=MUTPB, mu_mean=0, mu_sigma=1, grid=args.grid, graph=args.graph)
        if args.savefile is not None:
            row = [best_fitness, expr[0], expr[1], expr[2], "\r"]
            save_data(row, "../results/"+args.savefile)