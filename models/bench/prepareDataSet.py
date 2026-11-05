import random

from sklearn.datasets import load_iris

def startifiedSplitDataSet():
    features = load_iris().data.tolist()


    target = load_iris().target.tolist()

    for i in range(len(features)):
        if target[i] == 0:
            target[i] = 1
        else:
            target[i] = 0

    X_train, Y_train, X_test, Y_test = [], [], [], []

    class1 = [i for i in range(len(features)) if target[i] == 0]
    class2 = [i for i in range(len(features)) if target[i] == 1]

    random.shuffle(class1)
    random.shuffle(class2)

    p = 0.7
    split1 = int(p * len(class1))
    split2 = int(p * len(class2))

    for index in class1[:split1]:
        X_train.append(features[index])
        Y_train.append(target[index])

    for index in class2[:split2]:
        X_train.append(features[index])
        Y_train.append(target[index])

    for index in class1[split1:]:
        X_test.append(features[index])
        Y_test.append(target[index])

    for index in class2[split2:]:
        X_test.append(features[index])
        Y_test.append(target[index])


    return X_train,Y_train,X_test, Y_test