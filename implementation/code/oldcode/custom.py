import multiprocessing
import numpy
import random
import math
import argparse as ap
from deap_custom import eaSimpleEarlyStop
from deap import base
from deap import creator
from deap import tools

from refmap import RefMap
from scanreader import Scan
from util import hausdorff, applytuple, graph_results, total_sum, save_data, evaluate_solution
from tqdm import trange

NGEN = 100
POP = 100
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

def main():
    global scanName
    global refmap
    global errorscan
    parser = ap.ArgumentParser(description="My Script")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--multicore", action='store_true')
    # parser.add_argument("--seed")
    parser.add_argument("--savefile", type=str, default="temp.csv")
    parser.add_argument("-v", action='store_true', default=False)
    parser.add_argument("--graph", action='store_true', default=False)
    # parser.add_argument("--max_gen", type=int)
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gen", type=int, default=200)


    args, leftovers = parser.parse_known_args()

    NGEN = args.gen
    POP = args.pop

    
    # Using full map
    refmap = RefMap("data/combined.csv", tolerance=args.tolerance).points

    # Using error
    # refmap = Scan(scanName)
    # refmap = applytuple(refmap.scan_points, refmap.posx, refmap.posy, refmap.rot)
    errorscan = Scan(scanName, tolerance=args.tolerance)


#     graph_results(refmap, errorscan.scan_points, (-4.0466112179, 5.2727216101,-0.4485475159
# ))

    print "Aiming for"
    print errorscan.posx, errorscan.posy, errorscan.rot

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
    if args.multicore:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "max", "min", "std"

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("std", numpy.std)
    

    print "Starting"
    for x in trange(args.iterations):
        random.seed()
        record, log = eaSimpleEarlyStop(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                       stats=stats, halloffame=hof, verbose=args.v, stopval=0.1)
        if args.savefile is not None:
            row = [result[0], result[1], result[2], result[3], result[4], "\r"]
            save_data(row, "results/"+args.savefile)

        if args.v:
            print "Best individual:", expr
            print "Fitness:", evaluate_solution(expr[0], expr[1], expr[2], errorscan.posx, errorscan.posy, errorscan.rot)

        if args.graph:
            graph_results(refmap, errorscan.scan_points, expr)

if __name__ == "__main__":
    main()
