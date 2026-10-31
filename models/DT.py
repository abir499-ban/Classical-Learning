from utils import getFeaturesAndTargetFromIrisDataSet, splitData
import math

def majorityClass(classes):
    cnt = [0 for _ in range(3)]

    for c in classes:
        cnt[c] += 1 

    return cnt.index(max(cnt))


def getentropy(y):
    cnt = [0 for _ in range(3)]

    for c in y:
        cnt[c] += 1 

    entropy = 0
    for count in cnt:
        f = count / len(y)
        if f == 0: continue
        entropy += f * math.log2(f)

    return -entropy

def infoGain(X,Y, feature):
    groups = {}

    parentEntropy = getentropy(Y)

    for i,row in enumerate(X):
        if row[feature] not in groups:
            groups[row[feature]] = [] 
        
        groups[row[feature]].append(Y[i])

    weightedEntropy = 0.0

    for grp in groups:
        f = len(groups[grp]) / len(Y)
        weightedEntropy += f * getentropy(groups[grp])

    return parentEntropy - weightedEntropy

def buildTree(X,Y,features):


    classes = set(Y)
        
    if len(classes) == 1 or len(features) == 0:
        return {"class" : majorityClass(Y)}

    bestGain, bestFeature = 0,-1

    for f in features:
        ig = infoGain(X,Y,f)
        if ig > bestGain:
            bestGain = ig 
            bestFeature = f 

    if bestGain <= 0:
        return {"class": majorityClass(Y)}

    tree = {
        "feature" : bestFeature,
        "class" : majorityClass(Y),
        "branches" : {}
    }

    remFeatures = [f for f in features if f != bestFeature]
    groups = {}
    for i,row in enumerate(X):
        if row[bestFeature] not in groups:
            groups[row[bestFeature]] = [] 

        groups[row[bestFeature]].append(i)

    for grp,index in groups.items():
        newX = [X[i] for i in index]
        newY = [Y[i] for i in index]
        tree["branches"][grp] = buildTree(
            newX,
            newY,
            remFeatures
        )

    return tree   


def predictSample(sample, tree):

    if "feature" not in tree:
        return tree["class"]

    f = sample[tree["feature"]]

    if f not in tree["branches"]:
        return tree["class"]

    return predictSample(sample, tree["branches"][f])





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

    # Sepal Width
    if row[1] <= 2.8:
        new_row.append("Narrow")
    elif row[1] <= 3.4:
        new_row.append("Medium")
    else:
        new_row.append("Wide")

     # Petal Length
    if row[2] <= 2.5:
        new_row.append("Short")
    elif row[2] <= 4.9:
        new_row.append("Medium")
    else:
         new_row.append("Long")

    # Petal Width
    if row[3] <= 0.8:
        new_row.append("Narrow")
    elif row[3] <= 1.6:
        new_row.append("Medium")
    else:
        new_row.append("Wide")

    categoricalFeatures.append(new_row)


X_train, Y_train, X_test, Y_test = splitData(categoricalFeatures , target,0.7)
features = list(range(4))

tree= buildTree(X_train , Y_train, features)
correct = 0

for i,t in enumerate(X_test):
    prediction = predictSample(t,tree)
    if prediction == Y_test[i]: correct += 1

Acc = (correct * 100) /  len(Y_test)

print("Decision Tree Accuracy:", Acc)