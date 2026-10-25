import math

from matplotlib import markers

from dataset.iris import IrisDataMatrix
from logic.featureSelection.utils import TARGETINDX
import matplotlib.pyplot as plt

def euclideanDis(data1, data2):
    d = sum([(d1-d2)**2 for d1,d2 in zip(data1, data2)])
    return math.sqrt(d) 

def Gauss(d, sigma):
    p = math.exp(
        -(d**2) / (2*sigma**2)
    )

    return p 

def predictClass(testTuple, trainData, sigma):

    classLabels = [0 for _ in range(3)]

    for trainTuple in trainData:
        dis = euclideanDis(
            testTuple[:4] , trainTuple[:4]
        )
        p = Gauss(dis, sigma)
        l = TARGETINDX[trainTuple[4]]
        classLabels[l] += p 

    maxLabel,maxP = -1,0
    for l,p in enumerate(classLabels):
        if p > maxP:
            maxLabel = l 
            maxP = p  

    return maxLabel

def runPNN(trainData, testData, sigma):
    acc = 0 

    for testTuple in testData:
        predLabel = predictClass(testTuple, trainData , sigma)
        if predLabel == TARGETINDX[testTuple[4]]: acc += 1 

    return (acc / len(testData)) * 100 


sigma_values = []
accuracy_scores = []

sigma = 0.05 

trainData, testData = [], []

for i in range(len(IrisDataMatrix)):
    if i % 5 == 0:
        testData.append(IrisDataMatrix[i])
    else:
        trainData.append(IrisDataMatrix[i])

while sigma <= 1.5:
    accuracy_scores.append(runPNN(trainData, testData, sigma))
    sigma_values.append(sigma)
    sigma += 0.05

bestAcc, bestSigma = 0,-1

for i in range(len(sigma_values)):
    if accuracy_scores[i] > bestAcc:
        bestAcc = accuracy_scores[i]
        bestSigma = sigma_values[i] 

print(f"PNN Results \n Best Accuracy : {bestAcc} \n Best Sigma : {bestSigma}")

plt.figure(figsize=(10,10))

plt.plot(
    sigma_values,
    accuracy_scores,
    marker="o",
)

plt.xlabel("Sigma Values")
plt.ylabel("Accuracy")
plt.title("PNN Accuracy vs Sigma")

plt.show()


