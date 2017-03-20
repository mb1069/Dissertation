#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generaly, the keyword used are
:meth:`mate` for crossover, :meth:`tr_mutate` for tr_mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random

from deap import tools
from deap import algorithms
from tqdm import trange
from operator import itemgetter
from util import hausdorff, applytuple, graph_results, total_sum, save_data, evaluate_solution, graph_gen, update_series, initPop


def eaSimpleEarlyStop(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, stopval=0.1):
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

        # Early stopping 
        maxfit = max(fitnesses, key=itemgetter(0))[0]
        if maxfit>=stopval:
            break


        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        desc = str(toolbox.evaluate(tools.selBest(population, 1)[0])[0])
        

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream
        else:
            pbar.set_description(desc)
    expr = tools.selBest(pop, 1)[0]
    return gen, toolbox.evaluate(expr)[0], expr[0], expr[1], expr[2]



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
        graph, pop_series = graph_gen(refmap, population, target)

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

def varOrZeno(population, toolbox, lambda_, cxpb, mutpb, progress, mu, sigma):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in xrange(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = tools.mutGaussian(ind, mu, sigma*max(progress,0.4), mutpb)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring

# def mutGaussianZeno(individual, mu, sigma, indpb):
#     """This function applies a gaussian mutation of mean *mu* and standard
#     deviation *sigma* on the input individual. This mutation expects a
#     :term:`sequence` individual composed of real valued attributes.
#     The *indpb* argument is the probability of each attribute to be mutated.
    
#     :param individual: Individual to be mutated.
#     :param mu: Mean or :term:`python:sequence` of means for the
#                gaussian addition mutation.
#     :param sigma: Standard deviation or :term:`python:sequence` of 
#                   standard deviations for the gaussian addition mutation.
#     :param indpb: Independent probability for each attribute to be mutated.
#     :returns: A tuple of one individual.
    
#     This function uses the :func:`~random.random` and :func:`~random.gauss`
#     functions from the python base :mod:`random` module.
#     """
#     size = len(individual)
#     if not isinstance(mu, Sequence):
#         mu = repeat(mu, size)
#     elif len(mu) < size:
#         raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
#     if not isinstance(sigma, Sequence):
#         sigma = repeat(sigma, size)
#     elif len(sigma) < size:
#         raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    
#     for i, m, s in zip(xrange(size), mu, sigma):
#         if random.random() < indpb:
#             individual[i] += random.gauss(m, s)
    
#     return individual,


def mutSplitGaussian(individual, tr_sigma, r_sigma, indpb):
    """This function applies a gaussian tr_mutation of mean *tr_mu* and standard
    deviation *tr_sigma* on the input individual. This tr_mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be tr_mutated.
    
    :param individual: Individual to be tr_mutated.
    :param tr_mu: Mean or :term:`python:sequence` of means for the
               gaussian addition tr_mutation.
    :param tr_sigma: Standard deviation or :term:`python:sequence` of 
                  standard deviations for the gaussian addition tr_mutation.
    :param indpb: Independent probability for each attribute to be tr_mutated.
    :returns: A tuple of one individual.
    
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    # size = len(individual)
    # if not isinstance(tr_mu, Sequence):
    #     tr_mu = repeat(tr_mu, size)
    # elif len(tr_mu) < size:
    #     raise IndexError("tr_mu tr_must be at least the size of individual: %d < %d" % (len(tr_mu), size))
    # if not isinstance(tr_sigma, Sequence):
    #     tr_sigma = repeat(tr_sigma, size)
    # elif len(tr_sigma) < size:
    #     raise IndexError("tr_sigma tr_must be at least the size of individual: %d < %d" % (len(tr_sigma), size))
    
    # for i, m, s in zip(xrange(size), tr_mu, tr_sigma):
    if random.random() < indpb:
        individual[0] += random.gauss(0, tr_sigma)    
    if random.random() < indpb:
        individual[1] += random.gauss(0, tr_sigma)
    if random.random() < indpb:
        individual[2] += random.gauss(0, r_sigma)
    
    return individual,