from copy import deepcopy
import random

from .constants import MUTATION_RATE, POPULATION_SIZE, FeaturesCount


def createPopulation():
    population = []

    while len(population) < POPULATION_SIZE:
        chromosome = []

        for j in range(4):
            chromosome.append(random.randint(0,1))

        if chromosome == [0,0,0,0]: continue
        population.append(chromosome)

    return population


def selectParents(chromosome, fitness):

    selections = []

    for _ in range(POPULATION_SIZE):
        a = random.randint(0, POPULATION_SIZE-1)
        b = random.randint(0, POPULATION_SIZE-1)
        c = random.randint(0, POPULATION_SIZE-1)

        bestFtinessIndex = a 

        if fitness[b] > fitness[bestFtinessIndex]:
            bestFtinessIndex = b 

        if fitness[c] > fitness[bestFtinessIndex]:
            bestFtinessIndex = c 

        selections.append(deepcopy(chromosome[bestFtinessIndex]))

    return selections

def crossover(parent1, parent2):
    crossoverPoint = random.randint(1, 3)

    child1 = parent1[:crossoverPoint] + parent2[crossoverPoint:]
    child2 = parent2[:crossoverPoint] + parent1[crossoverPoint:]

    return child1, child2

def mutate(chromosome):
    for j in range(FeaturesCount):
        if random.random() < MUTATION_RATE:
            chromosome[j] = 1 - chromosome[j]

    if chromosome == [0,0,0,0]:
        chromosome[random.randint(0,3)] = 1 

    return chromosome

def getworstFitnessIndex(fitness):
    worstFitnessIndex = 0

    for i in range(len(fitness)):
        if fitness[i] < fitness[worstFitnessIndex]:
            worstFitnessIndex = i 

    return worstFitnessIndex

