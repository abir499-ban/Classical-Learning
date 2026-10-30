

import math
import matplotlib.pyplot as plt
import random


from utils import getFeaturesAndTargetFromIrisDataSet, splitData


def dot(vec1, vec2):
    return sum([d1*d2 for d1,d2 in zip(vec1, vec2)])

def normalizeFeature(mat):
    for j in range(len(mat[0])):
        colMax, colMin = float('-inf'), float('inf')
        for i in range(len(mat)):
            colMax = max(colMax ,mat[i][j])
            colMin = min(colMin, mat[i][j])

        for i in range(len(mat)):
            mat[i][j] = (mat[i][j] - colMin) / (colMax - colMin) if colMax!=colMin else 0.0
    
    

def activationFunction(x):
    x = max(-500 , min(500 , x))
    fx = 1.0/(1.0+math.exp(-x))
    return fx


def buildNetwork(featureCount , hiddenNeuronCount , outputNeuronCount):
    network = []

    hidden = []
    for _ in range(hiddenNeuronCount):
        weights = [random.uniform(-0.5,0.5) for _ in range(featureCount)]
        bias = random.uniform(-0.5,0.5)
        v = {
            'w' : weights,
            'b' : bias
        }
        hidden.append(v)

    output = []

    for _ in range(outputNeuronCount):
        weights = [random.uniform(-0.5,0.5) for _ in range(hiddenNeuronCount)]
        bias = random.uniform(-0.5,0.5)
        v = {
            'w' : weights,
            'b' : bias
        }
        output.append(v)

    network.append(hidden)
    network.append(output)

    return network

def forward(network, sample):

    hiddenActivations = []

    for neuron in network[0]:
        yIn = dot(neuron['w'] , sample) + neuron['b']
        yOut = activationFunction(yIn)
        hiddenActivations.append(yOut)

    outputLayerActivations = []

    for neuron in network[1]:
        zIn = dot(neuron['w'] , hiddenActivations) + neuron['b']
        zOut = activationFunction(zIn)
        outputLayerActivations.append(zOut)

    return hiddenActivations, outputLayerActivations 


def backward(network, sample , target, learningRate , Momentum):
    hiddenActivations, opActivations = forward(network , sample)

    opdeltas = []
    error = 0.0
    for i in range(len(network[1])):
        diff = (target[i] - opActivations[i])
        error += diff ** 2
        delta = diff * opActivations[i] * (1-opActivations[i])
        opdeltas.append(delta)

    hiddenDeltas = []
    for i in range(len(network[0])):
        e = 0.0
        for j,opNeuron in enumerate(network[1]):
            e += opdeltas[j] * opNeuron['w'][i]

        delta = e * hiddenActivations[i] * (1-hiddenActivations[i])
        hiddenDeltas.append(delta)

    for i,neuron in enumerate(network[1]):
        for k in range(len(network[0])):
            neuron['w'][k] += learningRate * opdeltas[i] * hiddenActivations[k]

        neuron['b'] +=  learningRate * opdeltas[i]

    for i,neuron in enumerate(network[0]):
            for k in range(len(sample)):
                neuron['w'][k] += learningRate * hiddenDeltas[i] * sample[k]
    
            neuron['b'] += learningRate * hiddenDeltas[i]

    return error / len(network[1])


def train(network , X_train, Y_train):
    learning_rate, momentum = 0.5,1
    epochs = 500

    avgError = []

    for _ in range(epochs):
        e = 0.0
        for x,y in zip(X_train , Y_train):
            target = [0,0,0]
            target[y] = 1
            e += backward(network , x,target,learning_rate , momentum)
        avgError.append(e / len(X_train))

    plt.figure(figsize=(10,10))
    plt.plot(
        [(i+1) for i in range(len(avgError))],
        avgError,
        marker='o'
    )

    plt.xlabel("Epochs")
    plt.ylabel("Average Error")
    plt.title("Epochs vs Error")

    plt.show()

def predictSample(network, sample):
    _,opActivations = forward(network, sample)

    return opActivations.index(max(opActivations))



feature,target = getFeaturesAndTargetFromIrisDataSet()
normalizeFeature(feature)
X_train, Y_train, X_test, Y_test = splitData(feature , target, 0.7)


network = buildNetwork(4,6,3)

train(network , X_train, Y_train)

correct = 0

for i,sample in enumerate(X_test):
    predictedClass = predictSample(network , sample)
    if predictedClass == Y_test[i]: correct += 1


Accuracy = (correct * 100) / len(X_test)

print("Average Accurancy : ",Accuracy)
print(f'Correct Sample : {correct} out of {len(X_test)}')












    

    



    