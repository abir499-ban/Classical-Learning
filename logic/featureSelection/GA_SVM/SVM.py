
from sklearn.svm import SVC

from .constants import FeaturesCount, SampleCount


def trainSVM(X_train, Y_train, X_test):
    model = SVC(kernel='linear')
    model.fit(X_train , Y_train)

    predictions = model.predict(X_test)

    return predictions
    

def calculateFitness(chromosome , X_train, X_test, Y_train, Y_test):
    selected_Xtrain, selected_Xtest = [],[]

    selected_features = []

    for i in range(FeaturesCount):
        if chromosome[i] == 1: selected_features.append(i)

    for x_train in X_train:
        r = []
        for i in selected_features:
            r.append(x_train[i])
        selected_Xtrain.append(r)

    for x_test in X_test:
        r = []
        for i in selected_features:
            r.append(x_test[i])
        selected_Xtest.append(r)

    predictions = trainSVM(selected_Xtrain, Y_train, selected_Xtest)

    correct = 0

    for i in range(len(Y_test)):
        if predictions[i] == Y_test[i]: correct += 1

    return correct / len(Y_test)

    

    
    

