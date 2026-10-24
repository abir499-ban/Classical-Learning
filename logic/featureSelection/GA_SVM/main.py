
from logic.featureSelection.GA_SVM.data_loader import loadFeatureAndTarget

from .GA import createPopulation, crossover, getworstFitnessIndex, mutate, selectParents
from .SVM import calculateFitness
from .constants import GENERATIONS, POPULATION_SIZE
from .data_loader import train_test_split
from copy import deepcopy
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split


population = createPopulation()
trainX, trainY, testX, testY = train_test_split(0.3, True)
# X, target = loadFeatureAndTarget()
# trainX, testX, trainY, testY = train_test_split(
#     X,
#     target,
#     test_size=0.30,
#     random_state=42,
#     stratify=target
# )

bestGlobalFitness = 0
bestGlobalFitnessChoromosome=  None 
avgFitnessHistory = []
bestFitnessHistory = [] 
generations = [i+1 for i in range(GENERATIONS)]

for g in range(GENERATIONS):
    fitness = []

    for chromosome in population:
        f = calculateFitness(chromosome, trainX, testX, trainY, testY)
        fitness.append(f)

    generationBestFitness , generationBestChormosome = 0 , None 
    Sf = 0
    for i,f in enumerate(fitness):
        Sf += f 
        if f > generationBestFitness:
            generationBestFitness = f
            generationBestChormosome = deepcopy(population[i])
    avgFitness  = Sf / len(fitness)
    avgFitnessHistory.append(avgFitness)

    if generationBestFitness > bestGlobalFitness:
        bestGlobalFitness = generationBestFitness 
        bestGlobalFitnessChoromosome = deepcopy(generationBestChormosome)

    print(
        f"Generation {g+1} |    Avg Fitness : {avgFitness} |  BestFitness : {bestGlobalFitness}"
    )

    eliteChormosome = deepcopy(generationBestChormosome)

    selectedParents = selectParents(population, fitness)

    newGeneration = []

    for i in range(0, POPULATION_SIZE , 2):
        parent1, parent2 = selectedParents[i] , selectedParents[i+1]
        child1, child2 = crossover(parent1, parent2)

        newGeneration.append(child1)
        newGeneration.append(child2)

    for i in range(len(newGeneration)):
        newGeneration[i] = mutate(newGeneration[i])

    newGenerationFitness = []

    for chr in newGeneration:
        newGenerationFitness.append(calculateFitness(chr , trainX, testX, trainY, testY))

    worstFitnessIndex = getworstFitnessIndex(newGenerationFitness)
    newGeneration[worstFitnessIndex] = eliteChormosome

    population = newGeneration

print("FINAL RESULT")
print(f"Best Fitness : {bestGlobalFitness} | Best Chromosome : {bestGlobalFitnessChoromosome}")

plt.figure(figsize=(10,10))

plt.plot(
    generations,
    avgFitnessHistory,
    marker="o",
    linewidth=2,
    markersize=6
)


plt.xlabel(
    "Generation",
    fontsize=12
)

plt.ylabel(
    "Average Fitness",
    fontsize=12
)


plt.title(
    "GA-Based Feature Selection using SVM",
    fontsize=15
)


# plt.xticks(generations)


# plt.ylim(0.80, 1.02)


# plt.grid(
#     True,
#     alpha=0.3
# )

# plt.tight_layout()
plt.show()





    