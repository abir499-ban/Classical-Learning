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

