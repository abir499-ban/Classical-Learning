import math

from dataset.iris import IrisDataMatrix
from logic.featureSelection.utils import TARGETINDX
import matplotlib.pyplot as plt

def EucledeanDis(data1, data2):
    dis = sum([(d1-d2)**2 for d1, d2 in zip(data1, data2)])

    return math.sqrt(dis)


def runKNN(trainData, testTuple , k):

    distances = []

    for trainTuple in trainData:
        dis = EucledeanDis(
            trainTuple[:4] , testTuple[:4]
        )
        distances.append((dis, TARGETINDX[trainTuple[4]]))

    distances.sort(key=lambda x : x[0])

    topKDis = distances[:k]

    classLabels = [0 for _ in range(3)]

    for dis, label in topKDis:
        classLabels[label] += 1 

    predictedClass, maxVotes = -1,0

    for label, vote in enumerate(classLabels):
        if vote > maxVotes:
            maxVotes = vote 
            predictedClass = label 

    return predictedClass


def getAccuracy(trainData, testData, k):
    correct = 0

    for testTuple in testData:
        predictedLabel = runKNN(trainData , testTuple , k)
        actualLabel = TARGETINDX[testTuple[4]]
        if actualLabel == predictedLabel: correct += 1 

    return correct / len(testData) * 100


kValues = []
accuracyScores = []
trainData, testData = [], []

for i in range(len(IrisDataMatrix)):
    if i % 5 == 0:
        testData.append(IrisDataMatrix[i])
    else:
        trainData.append(IrisDataMatrix[i])

for k in range(1,31):
    accuracyScores.append(getAccuracy(trainData, testData, k))
    kValues.append(k) 

bestAccuracy , bestK = 0,-1
for i in range(len(accuracyScores)):
    if accuracyScores[i] > bestAccuracy:
        bestAccuracy = accuracyScores[i] 
        bestK = kValues[i]


print(
    f"KNN Results \n Best Accuracy : {bestAccuracy} | K Value : {bestK}"
)


plt.figure(figsize=(10,10))

plt.plot(
    kValues,
    accuracyScores,
    marker='o',
)

plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K value")
plt.show()
