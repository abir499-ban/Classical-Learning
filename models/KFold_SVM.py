from sklearn.svm import SVC

from utils import getFeaturesAndTargetFromIrisDataSet

def genereateKFolds(features , target , k):
    folds = [[[],[]] for _ in range(k)]
    classIndices = [[] for _ in range(3)]

    for classLabel in range(3):
        for i in range(len(features)):
            if target[i] == classLabel:
                classIndices[classLabel].append(i)

    for classLabel in range(3):
        for i in range(len(classIndices[classLabel])):
            foldIndex = i % k
            tupleIndex = classIndices[classLabel][i]
            folds[foldIndex][0].append(features[tupleIndex])
            folds[foldIndex][1].append(target[tupleIndex])


    return folds



def runSVM(X_train, Y_train, X_test, Y_test):
    model = SVC(kernel='linear')
    model.fit(X_train, Y_train)
    predictedClasses = model.predict(X_test)

    correct = 0

    for i,l in enumerate(predictedClasses):
        if l == Y_test[i] : correct += 1 

    return (correct / len(Y_test)) * 100 


features,target = getFeaturesAndTargetFromIrisDataSet()
K = 5
KFoldData = genereateKFolds(features , target , K)

accuracy = [0 for _ in range(K)]
avgAccuracy , bestAcc = 0.0 , 0

for i in range(K):
    testFeatures , testTarget = KFoldData[i]
    trainFeatures, trainTarget = [] ,[]

    for j in range(K):
        if j == i: continue 
        trainFeatures += KFoldData[j][0]
        trainTarget += KFoldData[j][1]

    accuracy[i] = runSVM(trainFeatures , trainTarget , testFeatures, testTarget)
    bestAcc = max(bestAcc , accuracy[i])
avgAccuracy = sum(accuracy) / K

print("K FOLD SVM CLASSIFICATION RESULT")
print(f"Accuracy results ")

for i,acc in enumerate(accuracy):
    print(f"Fold : {i+1} | Accurancy : {acc}")
print(f"Best Accuracy : {bestAcc} | AvgAccuracy : {avgAccuracy}")



        
                