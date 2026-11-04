


from copy import deepcopy
import math
import random


from utils import getFeaturesAndTargetFromIrisDataSet

def prepareDataSet():
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


    initialWeights = [1/len(categoricalFeatures) for _ in range(len(categoricalFeatures))]

    return categoricalFeatures , target, initialWeights

def majority(Y):
    cnt = [0] * 3
    for y in Y:
        cnt[y] += 1

    return cnt.index(max(cnt))

def findEntropy(Y):
    entropy = 0.0
    c = [0 for _ in range(3)]

    for y in Y:
        c[y] += 1 

    for cnt in c:
        f = cnt / len(Y)
        if f==0: continue
        entropy += f * math.log2(f)

    return -entropy

def findWeightedEntropy(X,Y,feature):
    grps = {}

    for i,row in enumerate(X):
        if row[feature] not in grps:
            grps[row[feature]] = [] 
        grps[row[feature]].append(Y[i])

    we = 0.0
    for grp in grps:
        f = len(grps[grp]) / len(X)
        we += f*findEntropy(grps[grp])

    return we 

def createStump(X,Y,feature):
    stump = {}
    grps = {}

    for i,row in enumerate(X):
        if row[feature] not in grps:
            grps[row[feature]] = [] 
        grps[row[feature]].append(Y[i])

    for grp in grps:
        stump[grp] = majority(grps[grp])

    return stump

def predictSample(sample, stump, feature,Y):
    f = sample[feature]
    return stump.get(f,majority(Y))

def trainBaseModel(X,Y, trainX, trainY):
    bestFeature, lowE = -1, float('inf')

    for j in range(len(X[0])):
        e = findWeightedEntropy(X,Y,j)
        if e < lowE:
            lowE = e 
            bestFeature = j 

    stump = createStump(X,Y,bestFeature)
    corrected,misclasfied = [],[]

    for i in range(len(trainX)):
        predictedClass = predictSample(trainX[i], stump,bestFeature,trainY)
        if predictedClass == trainY[i]:
            corrected.append(i)
        else:
            misclasfied.append(i)

    return corrected , misclasfied,stump,bestFeature

def runAdaboost(X,Y, weights, trainX, trainY):
    correctClassified, misClassified,stump,bestFeature = trainBaseModel(X,Y, trainX, trainY)
    TE = sum([weights[i] for i in misClassified]) 
    if TE >= 2/3:
        return trainX,trainY,weights
    if TE==0:
            models.append([stump, 10, bestFeature])
            return trainX,trainY,weights
    ## alpha = 0.5 * math.log((1-TE) / TE) // for binary classes
    alpha = math.log(2*(1-TE) / TE)  ## for multi class 

    

    ## update weights for correct records in binary

    # for i in correctClassified:
    #     weights[i] *= math.exp(-alpha)

    for i in misClassified:
        weights[i] *= math.exp(alpha)

    S = sum(weights)
    weights = [w/S for w in weights]

    newX,newY = [],[]
    for _ in range(len(trainX)):
        r = random.random()
        c = 0.000
        for j in range(len(trainX)):
            c += weights[j]
            if r <= c:
                newX.append(trainX[j])
                newY.append(trainY[j])
                break 
    models.append([stump, alpha, bestFeature])
    return newX, newY,weights 


features, target, weights = prepareDataSet()

X,Y = deepcopy(features) , deepcopy(target)

predictions = []
models = []

for _ in range(20):
    newX, newY,updatedWeights = runAdaboost(X,Y,weights,features,target)
    X,Y,weights = newX, newY,updatedWeights


for i in range(len(features)):
    clasVotes = [0.0] * 3
    for stump,alpha,f in models:
        predictedClass = predictSample(features[i] , stump,f,target)
        clasVotes[predictedClass] += alpha

    predictions.append(clasVotes.index(max(clasVotes)))

correct = sum([1 for i in range(len(features)) if predictions[i] == target[i]])
print(f"Accuracy: {(correct*100) / len(features)}")




    




