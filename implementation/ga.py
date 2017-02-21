import multiprocessing
import numpy
import random


import algorithms
from deap import base
from deap import creator
from deap import tools

from refmap import RefMap
from scanreader import Scan
from util import hausdorff, applytuple

NGEN = 1000
POP = 100
CXPB = 0.0
MUTPB = 1.0

MIN = -10
MAX = 10

Xerr = -3
Yerr = 2
rotErr = 0.1

refmap = RefMap("data/combined.csv", samplesize=1000)
errorscan = Scan("data/scan0").scan_points


def randfloat():
    return random.uniform(-5, 5)


def evaluate(individual):
    dataset = applytuple(errorscan, *individual)
    return hausdorff(dataset, refmap.points),


def check_bounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator


def main():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", randfloat)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.125, indpb=MUTPB)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint, indpb=CXPB)
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
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    print "Starting"
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                   stats=stats, halloffame=hof, verbose=True)
    expr = tools.selBest(pop, 1)[0]
    print expr


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
