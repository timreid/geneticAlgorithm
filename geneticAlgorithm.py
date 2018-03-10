"""
a simple yet flexible implementation of the classic genetic optimization algorithm.
it operates by maintaining a list of solutions, known as a population, beginning with a supplied inital population which is repeatedly reproduced by selecting individuals for reproduction in a manner analagous to natural selection and recombining them into new solution in a manner analagous to genetic crossover.
"""

from collections import namedtuple
import numpy

#an individual is a representation of a solution to the problem, consisting of a genome which is the solution itself, and its fitness score which is the quality of the solution.
Individual = namedtuple("Individual", "fitness genome")

def geneticAlgorithm(monitor, select, fitness, crossover, mutation, initialState, initialPopulation):
    """
    optimizes the supplied inital population by repeatedly generating new populations until a goal condition is met.
    Args:
        monitor: responsible for logging the progress and deciding when a goal condition has been met.
        select: chooses two parents for reproduction.
        fitness: provides a score for a genome.
        crossover: constructs a new genome from two parents genomes.
        mutation: applies a slight change to a genome with some small probability.
        initialState: the starting state of the run.
        initialPopulation: is the starting population of the run.
    """

    if not initialState["generation"]: initialState["generation"] = 0
    state = initialState
    population = initialPopulation

    #repeatedly generate new populations until the monitor sets the state to False
    while state:
        #the best solution, known as the elite, is automatically carried into the next generation thereby making the optimization monotonic.
        newPopulation = [population[0]]
        
        #construct the next population by selecting parents from the current population and crossing them over to create children
        while len(newPopulation) < len(initialPopulation):
            mom = select(state, population)
            dad = select(state, population)
            childGenome = crossover(state, mom.genome, dad.genome)
            mutation(state, childGenome)
            childFitness = fitness(state, childGenome)
            child = Individual(childFitness, childGenome)
            newPopulation.append(child)
        #populations are sorted by descending fitness
        newPopulation.sort(key=lambda individual: individual.fitness, reverse=True)
        population = newPopulation
        #monitor will perform any desired logging and set the state to False to indicate when a goal condition is reached.
        monitor(state, population)

def makeSimpleMonitor(outputFilename, genomeWriter):
    """
    returns a basic monitor which logs the generation number, the elite's fitness, and the elite's pretty printed genome. it sets the state to False when either a maximum number of generations or a sufficiently high elite fitness is reached.
    Args:
        outputFilename: the name given to the log file.
        genomeWriter: the genome pretty printer.
    """
    log = open(outputFilename, "w")
    def monitor(state, population):
        state["elite"] = population[0]
        state["generation"] += 1
        generation = state.get("generation")
        elite = state.get("elite")
        log.write(str(generation))
        log.write("\t")
        log.write(str(elite.fitness))
        log.write("\t")
        log.write(genomeWriter(elite.genome))
        log.write("\n")
        log.flush()
        maxGenerations = state.get("maxGenerations")
        goalFitness = state.get("goalFitness")
        #check if a goal condition has been met
        if (maxGenerations and generation >= maxGenerations) or (goalFitness and elite.fitness >= goalFitness):
            log.close()
            state = False
    return monitor


def simpleSelect(state, population):
    """
    selects an individual from the population using gaussian sampling.
    """
    n = len(population)
    sigma = n / 3
    #gaussian sampling with rejection
    i = int(abs(numpy.random.normal(0.0, sigma)))
    while i > n - 1:
        #sample is too big, try again
        i = int(abs(numpy.random.normal(0.0, sigma)))
    selected = population[i]
    return selected

def simpleCrossover(state, momGenome, dadGenome):
    """
    creates a child genome from two parents by cutting them at a random point and taking one slice from each.
    """
    #assumption: genomes are same length
    n = len(momGenome)
    crossoverPoint = numpy.random.random_integers(0, n - 1)
    childGenome = momGenome[0:crossoverPoint] + dadGenome[crossoverPoint:]
    return childGenome

def simpleBooleanMutation(state, genome):
    """
    randomize each codon in a genome with small probability
    """
    for codon in genome:
        if numpy.random.uniform(0.0, 1.0) <= state.get("mutationRate"):
            codon = numpy.random.choice([True, False])
