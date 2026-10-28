import numpy as np
from dataset.iris import IrisDataMatrix
from logic.featureSelection.utils import TARGETINDX

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
