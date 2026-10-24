from dataset.iris import IrisDataMatrix 
import random

def loadFeatureAndTarget():
    features = []
    target = []

    for i in range(len(IrisDataMatrix)):
        features.append(IrisDataMatrix[i][:4])
        target.append(IrisDataMatrix[i][4])

    return features, target


def train_test_split(split, shuffle):
    features , target = loadFeatureAndTarget()

    n = len(IrisDataMatrix)

    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)

    splitIndex = int(n * (1-split))

    trainX, testX, trainY, testY = [],[],[],[]

    for i in range(splitIndex):
        trainX.append(features[indices[i]])
        trainY.append(target[indices[i]])

    for i in range(splitIndex , n):
        testX.append(features[indices[i]])
        testY.append(target[indices[i]])

    return trainX, trainY, testX, testY