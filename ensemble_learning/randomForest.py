from copy import deepcopy
import random


from .decisionTree import buildTree,predictSample
from utils import getFeaturesAndTargetFromIrisDataSet, splitData

def categorizeDataset():
    features,target = getFeaturesAndTargetFromIrisDataSet()

    categoricalFeatures = []

    for row in features:
        new_row = []
        if row[0] <= 5.5:
            new_row.append("Short")
        elif row[0] <= 6.5:
            new_row.append("Medium")
        else:
            new_row.append("Long")

    
        if row[1] <= 2.8:
            new_row.append("Narrow")
        elif row[1] <= 3.4:
            new_row.append("Medium")
        else:
            new_row.append("Wide")

        if row[2] <= 2.5:
            new_row.append("Short")
        elif row[2] <= 4.9:
            new_row.append("Medium")
        else:
            new_row.append("Long")

        if row[3] <= 0.8:
            new_row.append("Narrow")
        elif row[3] <= 1.6:
            new_row.append("Medium")
        else:
            new_row.append("Wide")

        categoricalFeatures.append(new_row)

    return categoricalFeatures , target

def runRandomForest(X, Y):
    N = 20
    n = len(X[0])
    forest = []
    for _ in range(N):
        samplesFeatures = []
        sampleTarget = []
        for _ in range(30):
            i = random.randint(0,len(X)-1)
            samplesFeatures.append(X[i])
            sampleTarget.append(Y[i])

        indices = list(range(n))
        random.shuffle(indices)
        features = deepcopy(indices[:2])
        tree = buildTree(samplesFeatures, sampleTarget, features)
        forest.append(tree)

    return forest


if __name__=="__main__":
    features, target = categorizeDataset()
    X_train,Y_train,X_test,Y_test = splitData(features,target,0.7)
    forest = runRandomForest(X_train,Y_train)

    predictions = []
    
    for row in X_test:
        p = [0 for _ in range(3)]
        for tree in forest:
            pi = (predictSample(tree , row))
            p[pi] += 1
        predictions.append(p.index(max(p)))
    
    correct = 0
    for i in range(len(X_test)):
        if Y_test[i] == predictions[i]:
            correct += 1
    accuracy = (correct * 100) / len(X_test)    

    print(f"Random Forest : {accuracy}")