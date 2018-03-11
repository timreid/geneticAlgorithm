"""
a simple test application for the genetic algorithm.
the knapsack problem is to choose from a list of items, each with associated value and size, such that the value of the selection is maximized but does not exceed the size of the knapsack.
"""
import numpy
from geneticAlgorithm import *

def makeKnapsackMonitor(outputFilename, knapsackProblem):
	"""
	returns a basic monitor which logs the generation number, the elite's fitness, and the elite's pretty printed genome. it sets the state to False when either a maximum number of generations or a sufficiently high elite fitness is reached.
	Args:
		outputFilename: the name given to the log file.
		genomeWriter: the genome pretty printer.
	"""
	def knapsackGenomeWriter(genome):
		"""
		pretty prints a knapsack genome
		"""
		output = ""
		for codon in genome:
			if codon: output += "1"
			else: output += "0"
		return output

	def calculateWeight(genome):
		weight = 0.0
		for i in range(len(genome)):
			weight += genome[i] * knapsackProblem[i][0]
		return weight

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
		log.write(str(calculateWeight(elite.genome)))
		log.write("\t")
		log.write(knapsackGenomeWriter(elite.genome))
		log.write("\n")
		log.flush()
		maxGenerations = state.get("maxGenerations")
		goalFitness = state.get("goalFitness")
		#check if a goal condition has been met
		if (maxGenerations and generation >= maxGenerations) or (goalFitness and elite.fitness >= goalFitness):
			log.close()
			state = False
	return monitor

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
knapsackProblemSize = 10000
knapsackProblem = makeKnapsackProblem(knapsackProblemSize, 100.0, 100.0)
print(knapsackProblem)
maxWeight = 1000
monitor = makeKnapsackMonitor("output", knapsackProblem)
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