import math

from dataset.iris import IrisDataMatrix
from logic.featureSelection.utils import  CLASS_NAMES, TARGETINDX

X_train, Y_train, X_test, Y_test =[],[],[],[]

for i in range(len(IrisDataMatrix)):
    if i % 5 == 0:
        X_test.append(IrisDataMatrix[i][:4])
        Y_test.append(TARGETINDX[IrisDataMatrix[i][4]])
    else:
        X_train.append(IrisDataMatrix[i][:4])
        Y_train.append(TARGETINDX[IrisDataMatrix[i][4]])

def getPriorProbabilities():
    priors = [0 for _ in range(3)]

    for i in range(len(Y_train)):
        priors[Y_train[i]] += 1 

    for i in range(len(priors)):
        priors[i] /= len(Y_train)

    return priors

def getMean(data):
    mean = sum(data) / len(data)
    return mean 

def getVariance(data):
    mean = getMean(data)
    var = sum([(x-mean) ** 2 for x in data]) / len(data)
    return var 


def getGaussian(x,mean,variance):
    return -0.5*math.log(2*math.pi * variance) - ((x-mean) **2 / (2*variance))


def getStatistics(X_train, Y_train):
    statistics = {}

    for classLabel in range(3):
        for featureLabel in range(4):
            featureData = [X_train[i][featureLabel] for i in range(len(X_train)) if Y_train[i] == classLabel]
            mean = getMean(featureData)
            variance = getVariance(featureData)
            statistics[(classLabel, featureLabel)] = [mean, variance]

    return statistics 


def predictSample(testTuple, statistics, priors):
    probabilities = [math.log(priors[i]) for i in range(len(priors))]

    for classLabel in range(3):
        for featureLabel in range(4):
            x = testTuple[featureLabel]
            mean, variance = statistics[(classLabel, featureLabel)]
            probabilities[classLabel] += getGaussian(x,mean,variance)

    predictedClass, maxProb = -1,float('-inf')

    for classLabel, prob in enumerate(probabilities):
        if prob > maxProb:
            maxProb = prob 
            predictedClass = classLabel 

    return predictedClass 


def runNaiveBayes(X_train, Y_train, X_test, Y_test):
    prior = getPriorProbabilities()
    print("Prior Probabilities : ")

    for i,p in enumerate(prior):
        print(f"Class {CLASS_NAMES[i]}: {p}")

    statistics = getStatistics(X_train, Y_train)

    correct = 0

    for i,testTuple in enumerate(X_test):
        predictedClass = predictSample(testTuple, statistics, prior)
        print(f"Predicted Class for Sample {testTuple} : {CLASS_NAMES[predictedClass]} | Actual Class : {CLASS_NAMES[Y_test[i]]}")
        if predictedClass == Y_test[i]: correct += 1 

    acc = correct / len(Y_test)

    return acc 



accuracy_score = runNaiveBayes(X_train, Y_train, X_test, Y_test)
print(f"Naive Bayes Classifier Accuracy : {accuracy_score}")