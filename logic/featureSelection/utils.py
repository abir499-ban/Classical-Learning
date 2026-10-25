from dataset.iris import IrisDataMatrix

TAREGT_NAMES = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]

TARGETINDX = {
    "Iris-setosa" : 0,
    "Iris-versicolor" : 1,
    "Iris-virginica" : 2
}

FEATURE_NAME={
    0 : "Sepal Length",
    1 : "Sepal Width",
    2 : "Petal Length",
    3 : "Petal Width"
}



def getFeatures():
    features = []

    for j in range(4):
        r = []
        for i in range(len(IrisDataMatrix)):
            r.append(IrisDataMatrix[i][j])

        features.append(r)

    return features
    

def getTarget():
    target = []
    for i in range(len(IrisDataMatrix)):
        target.append(TARGETINDX[IrisDataMatrix[i][4]])

    return target 

def makeBins(data, num_bins):
    maxi = max(data)
    mini = min(data)

    w = (maxi - mini) / num_bins

    feature_bins = []

    for x in data:
        if x == maxi:
            feature_bins.append(num_bins - 1)
        else:
            feature_bins.append(int((x - mini) / w))

    return feature_bins

