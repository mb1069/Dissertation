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
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param tr_mutpb: The probability of tr_mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population and a :class:`~deap.tools.Logbook`
              with the statistics of the evolution.

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution (if
    any). The logbook will contain the generation number, the number of
    evalutions for each generation and the statistics if a
    :class:`~deap.tools.Statistics` if any. The *cxpb* and *tr_mutpb* arguments
    are passed to the :func:`varAnd` function. The pseudocode goes as follow
    ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, tr_mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select tr_multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.tr_mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
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

    return record, logbook


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