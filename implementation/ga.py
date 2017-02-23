import multiprocessing
import numpy
import random
import math

import algorithms
from deap import base
from deap import creator
from deap import tools

from refmap import RefMap
from scanreader import Scan
from util import hausdorff, applytuple, graph_results, total_sum

NGEN = 150
POP = 100
CXPB = 0.15
MUTPB = 0.05

MIN = -10
MAX = 10

TRANS_MIN, TRANS_MAX = -4.0, 4.0
ROT_MIN, ROT_MAX = 0, math.pi


scanName = "scans/scan110"
# Using full map
refmap = RefMap("data/combined.csv", tolerance=0.2).points

# Using error
# refmap = Scan(scanName)
# refmap = applytuple(refmap.scan_points, refmap.posx, refmap.posy, refmap.rot)
errorscan = Scan(scanName)

print "Aiming for"
print errorscan.posx, errorscan.posy, errorscan.rot


# errorscan.scan_points.append((-2.184327,2.641909))
# graph_results(refmap, errorscan.scan_points, (-2.184327,2.641909,1.1352500))
# graph_results(refmap, errorscan.scan_points, (0,0,0))


def evaluate(individual):
    dataset = applytuple(errorscan.scan_points, *individual)
    return 1/(1+total_sum(dataset, refmap)),
    # return 1/(1+hausdorff(dataset, refmap)),

def main():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_trans", random.uniform, TRANS_MIN, TRANS_MAX)
    toolbox.register("attr_rot", random.uniform, ROT_MIN, ROT_MAX)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_trans, toolbox.attr_trans, toolbox.attr_rot), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.125/4, indpb=MUTPB)
    # toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=MUTPB)

    toolbox.register("select", tools.selRoulette)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("evaluate", evaluate)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "max", "min", "std"

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("max", numpy.max)

    print "Starting"
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                   stats=stats, halloffame=hof, verbose=True)
    expr = tools.selBest(pop, 1)[0]
    print "Aiming for"
    print (errorscan.posx, errorscan.posy, errorscan.rot)
    print "Result:"
    print expr
    graph_results(refmap, errorscan.scan_points, expr)
    # graph_results(refmap, errorscan.scan_points, (errorscan.posx, errorscan.posy, errorscan.rot))

    # print len(self.points), samplesize
    # plt.figure("X/Y")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.scatter([p[0] for p in self.points], [p[1] for p in self.points], s=2, marker='x')
    # plt.show()

if __name__ == "__main__":
    main()
