"""
a simple yet flexible implementation of the classic genetic optimization algorithm.
it operates by maintaining a list of solutions, known as a population, beginning with a
supplied inital population which is repeatedly reproduced by selecting individuals for
reproduction in a manner analagous to natural selection and recombining them into new
solution in a manner analagous to genetic crossover.
"""

from collections import namedtuple
import numpy

#an individual is a representation of a solution to the problem, consisting of a genome
#which is the solution itself, and its fitness score which is the quality of the solution.
Individual = namedtuple("Individual", "fitness genome")

def genetic_algorithm(monitor, select, fitness, crossover, mutation, initial_state, initial_population):
    """
    optimizes the supplied inital population by repeatedly generating new populations
    until a goal condition is met.
    Args:
        monitor: logs progress and decides when a goal condition has been met.
        select: chooses two parents for reproduction.
        fitness: provides a score for a genome.
        crossover: constructs a new genome from two parents genomes.
        mutation: applies a slight change to a genome with some small probability.
        initial_state: the starting state of the run.
        initial_population: is the starting population of the run.
    """

    if not "generation" in initial_state:
        initial_state["generation"] = 0
    state = initial_state
    population = initial_population

    #repeatedly generate new populations until the monitor sets the state to False
    while state:
        #the best solution, known as the elite, is automatically carried into the next generation
        #thereby making the optimization monotonic.
        new_population = [population[0]]
        #construct the next population by selecting parents from the current population and
        #crossing them over to create a child
        while len(new_population) < len(initial_population):
            mom = select(state, population)
            dad = select(state, population)
            child_genome = crossover(state, mom.genome, dad.genome)
            child_genome = mutation(state, child_genome)
            child_fitness = fitness(state, child_genome)
            child = Individual(child_fitness, child_genome)
            new_population.append(child)
        #populations are sorted by descending fitness
        new_population.sort(key=lambda individual: individual.fitness, reverse=True)
        population = new_population
        #monitor will perform any desired logging and set the state to False to indicate
        #when a goal condition is reached.
        state = monitor(state, population)

def make_simple_monitor(output_filename, genome_writer):
    """
    returns a basic monitor which logs the generation number, the elite's fitness, and the elite's
    pretty printed genome. it sets the state to False when either a maximum number of generations or
    a sufficiently high elite fitness is reached.
    Args:
        output_filename: the name given to the log file.
        genome_writer: the genome pretty printer.
    """
    log = open(output_filename, "w")
    def monitor(state, population):
        """
        log each generation, and check for goal conditions
        """
        state["elite"] = population[0]
        state["generation"] += 1
        generation = state.get("generation")
        elite = state.get("elite")
        log.write(str(generation))
        log.write("\t")
        log.write(str(elite.fitness))
        log.write("\t")
        log.write(genome_writer(elite.genome))
        log.write("\n")
        log.flush()
        max_generations = state.get("max_generations")
        goal_fitness = state.get("goal_fitness")
        #check if a goal condition has been met
        if (max_generations and generation >= max_generations) or (goal_fitness and elite.fitness >= goal_fitness):
            log.close()
            #setting state to False signifies the end of the run
            state = False
        return state
    return monitor


def simple_select(state, population):
    """
    selects an individual from the population using gaussian sampling.
    """
    sigma = len(population) / 3
    #gaussian sampling with rejection
    i = int(abs(numpy.random.normal(0.0, sigma)))
    while i > len(population) - 1:
        #sample was too big, reject it and try again
        i = int(abs(numpy.random.normal(0.0, sigma)))
    selected = population[i]
    return selected

def simple_crossover(state, mom_genome, dad_genome):
    """
    creates a child genome from two parents by cutting them at a random point and
    taking one slice from each.
    """
    #assumption: genomes are same length
    crossover_point = numpy.random.random_integers(0, len(mom_genome) - 1)
    child_genome = mom_genome[0:crossover_point] + dad_genome[crossover_point:]
    return child_genome
