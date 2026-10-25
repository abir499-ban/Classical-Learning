import math


from .utils import FEATURE_NAME, getFeatures, getTarget, makeBins


def createContigencyTable(feature_bins, target , num_bins):
    table =[[0 for _ in range(num_bins)] for _ in range(3)]

    for i in range(len(feature_bins)):
        table[target[i]][feature_bins[i]] += 1

    return table 

def getRowAndColSums(table):
    rowSums = [sum(row) for row in table]
    colSums = []

    for j in range(len(table[0])):
        s = 0
        for i in range(len(table)):
            s += table[i][j]
        colSums.append(s)

    return rowSums, colSums 

def getExepctedFrequencyTable(table):
    expectedTable = [[0 for _ in range(len(table[0]))] for _ in range(len(table))]
    rowSums, colSums = getRowAndColSums(table)
    total = sum(rowSums)

    for i in range(len(table)):
        for j in range(len(table[0])):
            expectedTable[i][j] = (rowSums[i] * colSums[j]) / total

    return expectedTable

def getChiSquare(data, target, f):
    featureBins = makeBins(data , 3)
    table = createContigencyTable(featureBins , target , 3)
    expected = getExepctedFrequencyTable(table)

    m,n = len(table) , len(table[0])

    chiSq = 0

    for i in range(m):
        for j in range(n):
            chiSq += ((table[i][j] - expected[i][j]) ** 2) / expected[i][j]

    print(f"Contigency Table for Feature : {FEATURE_NAME[i]}")
    for row in table:
        print(row)
    

    return chiSq




features = getFeatures()
target = getTarget()

scores = []

for i in range(4):
    scores.append(getChiSquare(features[i] , target, i))

sortedFeatures = sorted(range(4) , key=lambda x : scores[x] , reverse=True)

for f in sortedFeatures:
    print(f"Feature : {FEATURE_NAME[f]} | Chi-Square : {scores[f]}")



