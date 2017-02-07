import random
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

NGEN = 500
CXPB = 0.5
MUTPB = 0.2

MIN = 0
MAX = 1


def randInt():
    return random.randint(0, 1)


def evaluate(individual):
    block_vals = []
    for val in xrange(0, IND_SIZE, 3):
        s = sum(individual[val:val + 3])
        v = 0
        if s == 0:
            v = 0.9
        elif s == 1:
            v = 0.8
        elif s == 2:
            v = 0.0
        elif s == 3:
            v = 1.0
        block_vals.append(v)
    return sum(block_vals) / len(block_vals),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", randInt)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform)
toolbox.register("evaluate", evaluate)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)

logbook = tools.Logbook()
logbook.header = "gen", "avg", "max", "min", "std"


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


toolbox.decorate("mate", check_bounds(MIN, MAX))
toolbox.decorate("mutate", check_bounds(MIN, MAX))


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                   stats=stats, halloffame=hof, verbose=True)


if __name__ == "__main__":
    main()
