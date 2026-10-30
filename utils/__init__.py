import numpy as np
from dataset.iris import IrisDataMatrix
from logic.featureSelection.utils import TARGETINDX
import random

def encoder(x):
    return np.where(x == 0 , -1, 1)

def decoder(x):
    return np.where(x == -1, 0, 1)

def getFeaturesAndTargetFromIrisDataSet():
    f ,t = [],[]

    for i in range(len(IrisDataMatrix)):
        f.append(IrisDataMatrix[i][:4])
        t.append(TARGETINDX[IrisDataMatrix[i][4]])

    return f,t

def splitData(feature, target,p):
    N = len(feature)
    indices = [i for i in range(len(feature))]
    random.shuffle(indices)

    split = int(p * len(feature))

    X_train = [feature[indices[i]] for i in range(split)]
    Y_train = [target[indices[i]] for i in range(split)]

    X_test = [feature[indices[i]] for i in range(split,N)]
    Y_test = [target[indices[i]] for i in range(split,N)]

    return X_train, Y_train, X_test, Y_test
