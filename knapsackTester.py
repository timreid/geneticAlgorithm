"""
a simple test application for the genetic algorithm.
the knapsack problem is to choose from a list of items, each with associated value and size, such that the value of the selection is maximized but does not exceed the size of the knapsack.
"""
import numpy
from geneticAlgorithm import *

def makeKnapsackGenome(knapsackProblemSize):
    """
    makes a random knapsack genome
    """
    genome = []
    for _ in range(knapsackProblemSize):
        if numpy.random.uniform(0.0, 1.0) <= 1/knapsackProblemSize:
            genome.append(True)
        else:
            genome.append(False)
    return genome


def knapsackGenomeWriter(genome):
    """
    pretty prints a knapsack genome
    """
    output = ""
    for codon in genome:
        if codon: output += "1"
        else: output += "0"
    return output

def makeKnapsackProblem(nItems, weightSigma, valueSigma):
    """
    makes a random instance of the knapsack problem of the form:
    knapsackProblem = [(weight, value)]
    Args:
        nItems: the number of items
        weightSigma: the sigma for the distribution of item weights
        valueSigma: the sigma for the distribution of the item values
    """
    knapsackProblem = []
    for _ in range(nItems):
        weight = abs(numpy.random.normal(0.0, weightSigma))
        value = abs(numpy.random.normal(0.0, valueSigma))
        knapsackProblem.append((weight, value))
    return knapsackProblem
    
def makeKnapsackFitness(knapsackProblem, maxWeight):
    """
    returns a fitness function for a knapsack problem
    """
    def knapsackFitness(state, genome):
        weight = 0.0
        fitnessScore = 0.0
        for i in range(len(genome)):
            weight += genome[i] * knapsackProblem[i][0]
            fitnessScore += genome[i] * knapsackProblem[i][1]
        if weight > maxWeight: fitnessScore = 0.0
        return fitnessScore
    return knapsackFitness

#construct a random knapsack problem
knapsackProblemSize = 1000
knapsackProblem = makeKnapsackProblem(knapsackProblemSize, 10.0, 100.0)

maxWeight = 100
monitor = makeSimpleMonitor("output", knapsackGenomeWriter)
fitness = makeKnapsackFitness(knapsackProblem, maxWeight)
initialState = {"mutationRate":0.05}

#construct a random initial population
populationSize = 100
initialPopulation = []
for _ in range(populationSize):
    genome = makeKnapsackGenome(knapsackProblemSize)
    fitnessScore = fitness(initialState, genome)
    individual = Individual(fitnessScore, genome)
    initialPopulation.append(individual)

geneticAlgorithm(monitor, simpleSelect, fitness, simpleCrossover, simpleBooleanMutation, initialState, initialPopulation)