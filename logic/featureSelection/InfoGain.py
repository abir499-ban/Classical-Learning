import math
from collections import Counter
from logic.featureSelection.utils import FEATURE_NAME, getFeatures, getTarget, makeBins


def entropy(data):
    total = len(data)
    counts = {}

    for x in data:
        if x in counts: counts[x] += 1
        else: counts[x] = 1

    e = 0

    for x in counts.values():
        f = x/total
        e += f * math.log2(f)

    return -e 

target = getTarget()
features = getFeatures()
InfoD = entropy(target)

igScores = []

for j in range(4):
    featureBins = makeBins(features[j] , 3)
    countMapp = Counter(featureBins)
    InfoF = 0

    for b,c in countMapp.items():
        subTarget = [target[i] for i in range(len(target)) if featureBins[i] == b]
        e = entropy(subTarget)
        InfoF += (c/len(target)) * e

    igScores.append(InfoD - InfoF)


rankedFeatures = sorted(range(4) , key=lambda x : igScores[x] , reverse=True)

for f in rankedFeatures:
    print(f"Feature Name : {FEATURE_NAME[f]} | IG : {igScores[f]}")

    


