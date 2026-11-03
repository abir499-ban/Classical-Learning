import math


def majority(Y):
    cnt = [0] * 3

    for y in Y:
        cnt[y] += 1 

    return cnt.index(max(cnt))


def entropy(Y):
    classes = [0 for _ in range(3)]

    for f in Y:
        classes[f] += 1

    entropy = 0.0 
    for cnt in classes:
        f = cnt / len(Y) 
        if f == 0: continue
        entropy += f * math.log2(f)

    return -entropy 

def infoGain(X,Y,feature):
    parentEntropy = entropy(Y)

    grps = {}

    for i,row in enumerate(X):
        if row[feature] not in grps:
            grps[row[feature]] = [] 

        grps[row[feature]].append(Y[i])

    weightedEntropy = 0.0

    for grp in grps.keys():
        f = len(grps[grp]) / len(X)
        weightedEntropy += f * entropy(grps[grp])

    return parentEntropy - weightedEntropy 


def buildTree(X,Y,features):

    if len(set(Y)) == 1 or len(features)== 0:
        return {"class" : majority(Y)}

    bestInfoG , bestFeature = 0,-1

    for feature in features:
        infoD = infoGain(X,Y,feature)
        if infoD > bestInfoG:
            bestInfoG = infoD
            bestFeature=feature 

    if bestInfoG <= 0:
        return {"class" : majority(Y)}
    
    remFeatures = [f for f in features if f != bestFeature]

    tree = {
        "feature" : bestFeature,
        "class" : majority(Y),
        "branch" : {}
    }

    grps = {}
    for i,row in enumerate(X):
        if row[bestFeature] not in grps:
            grps[row[bestFeature]] = [] 

        grps[row[bestFeature]].append(i)

    for category,list in grps.items():
        newX = [X[i] for i in list]
        newY = [Y[i] for i in list]
        tree["branch"][category] = buildTree(newX, newY,remFeatures)

    return tree


def predictSample(tree , sample):
    if "feature" not in tree and "class" in tree:
        return tree["class"]

    f = sample[tree["feature"]]

    if f not in tree["branch"]:
        return tree["class"]

    return predictSample(tree["branch"][f],sample)
