"""
a simple test application for the genetic algorithm.
the knapsack problem is to choose from a list of items, each with associated value and size,
such that the value of the selection is maximized but does not exceed the size of the knapsack.
"""
import numpy

from genetic_algorithm import genetic_algorithm, Individual, simple_crossover, simple_select

def simple_boolean_mutation(state, genome):
    """
    randomize each codon in a genome with small probability
    """

    if numpy.random.uniform(0.0, 1.0) <= state.get("mutation_rate"):
        i = numpy.random.randint(0, len(genome))
        genome[i] = not genome[i]
    return genome

def make_knapsack_monitor(output_filename, knapsack_problem):
    """
    returns a basic monitor which logs the generation number, the elite's fitness, and
    the elite's pretty printed genome. it sets the state to False when either a
    maximum number of generations or a sufficiently high elite fitness is reached.
    Args:
        output_filename: the name given to the log file.
        knapsack_problem: the instance of the knapsack problem.
    """

    def calculate_weight(genome):
        """
        calculates the total weight of items described in the genome
        """
        weight = 0.0
        for i, codon in enumerate(genome):
            if codon:
                weight += knapsack_problem[i][0]
        return weight

    log = open(output_filename, "w")
    def monitor(state, population):
        """
        perform logging and check for goal condition
        """
        state["elite"] = population[0]
        state["generation"] += 1
        generation = state.get("generation")
        elite = state.get("elite")
        # log.write(str(generation))
        # log.write("\t")
        log.write(str(elite.fitness))
        log.write("\t")
        log.write(str(calculate_weight(elite.genome)))
        log.write("\t")
        log.write("".join('1' if codon else '0' for codon in elite.genome))
        log.write("\n")
        log.flush()
        max_generations = state.get("max_generations")
        goal_fitness = state.get("goal_fitness")
        #check if a goal condition has been met
        if (max_generations and generation >= max_generations) or (goal_fitness and elite.fitness >= goal_fitness):
            log.close()
            state = False
        return state
    return monitor

def make_knapsack_genome(knapsack_problem_size):
    """
    makes a random knapsack genome
    """
    genome = [False for i in range(knapsack_problem_size)]
    i = numpy.random.randint(0, knapsack_problem_size)
    genome[i] = True 
    return genome

def make_knapsack_problem(n_items, weight_sigma, value_sigma):
    """
    makes a random instance of the knapsack problem of the form:
    knapsack_problem = [(weight, value)]
    Args:
        n_items: the number of items
        weight_sigma: the sigma for the distribution of item weights
        value_sigma: the sigma for the distribution of the item values
    """
    knapsack_problem = []
    for _ in range(n_items):
        weight = abs(numpy.random.normal(0.0, weight_sigma))
        value = abs(numpy.random.normal(0.0, value_sigma))
        knapsack_problem.append((weight, value))
    return knapsack_problem
def make_knapsack_fitness(knapsack_problem, max_weight):
    """
    returns a fitness function for a knapsack problem
    """
    def knapsack_fitness(state, genome):
        """
        calculates the fitness score of the genome
        """
        weight = 0.0
        fitness_score = 0.0
        for i, codon in enumerate(genome):
            if codon:
                weight += knapsack_problem[i][0]
                fitness_score += knapsack_problem[i][1]
        if weight > max_weight:
            fitness_score = 0.0
        return fitness_score
    return knapsack_fitness

#construct a random knapsack problem
KNAPSACK_PROBLEM_SIZE = 1000
KNAPSACK_PROBLEM = make_knapsack_problem(KNAPSACK_PROBLEM_SIZE, 10.0, 100.0)
MAX_WEIGHT = 1000
MONITOR = make_knapsack_monitor("output", KNAPSACK_PROBLEM)
FITNESS = make_knapsack_fitness(KNAPSACK_PROBLEM, MAX_WEIGHT)
INITIAL_STATE = {"mutation_rate":0.05}
#construct a random initial population
POPULATION_SIZE = 100
INITIAL_POPULATION = []
for _ in range(POPULATION_SIZE):
    current_genome = make_knapsack_genome(KNAPSACK_PROBLEM_SIZE)
    current_fitness_score = FITNESS(INITIAL_STATE, current_genome)
    individual = Individual(current_fitness_score, current_genome)
    INITIAL_POPULATION.append(individual)

genetic_algorithm(MONITOR, simple_select, FITNESS, simple_crossover, simple_boolean_mutation, INITIAL_STATE, INITIAL_POPULATION)
