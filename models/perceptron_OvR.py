
import random

from utils import getFeaturesAndTargetFromIrisDataSet, splitData
import matplotlib.pyplot as plt

def dotProduct(vec1, vec2):

    return sum([v1*v2 for v1,v2 in zip(vec1,vec2)])


def activationFn(y , g):
    if y < -g: return -1
    elif y>g: return 1
    else: return 0 


def trainPerceptron(X_train , Y_train , targetClass, learningRate , g):

    weights = [0.0] * len(X_train[0])
    bias = 0.0
    epochs = 100
    avgError = []

    for _ in range(epochs):
        isError = False
        Error = 0
        for i in range(len(X_train)):
            actualClass = 1 if Y_train[i] == targetClass else -1 
            y_in = dotProduct(weights ,X_train[i]) + bias
            y_out = activationFn(y_in , g)
            if y_out == actualClass:
                continue

            isError = True
            Error += (actualClass - y_out) ** 2
            for j in range(len(weights)):
                weights[j] += learningRate * X_train[i][j] * actualClass

            bias += learningRate * actualClass

        avgError.append(Error / len(X_train))
        
        if not isError: break

    epochsIndex = [(i+1) for i in range(len(avgError))]
    plt.figure(figsize=(10,10))
    plt.plot(
        epochsIndex,
        avgError,
        marker='o'
    )
    plt.xlabel("Epochs")
    plt.ylabel("Average Accuracy")
    plt.title(f"Epoch vs Average Accuracy for Class {targetClass}")
    plt.show()

    return [weights , bias] 


def predictSample(sample, models):
    dps = [0 for _ in range(len(models))]
    for c, [w, b] in enumerate(models):
        dps[c] = dotProduct(w,sample) + b

    return dps.index(max(dps))


feature,target = getFeaturesAndTargetFromIrisDataSet()
X_train, Y_train, X_test, Y_test = splitData(feature , target, 0.7)
learning_rate = 0.1
g = 2
random.seed(42)


models = []

for classLabel in range(3):
    models.append(trainPerceptron(X_train , Y_train, classLabel , learning_rate , g))


predictions = [-1 for _ in range(len(X_test))]
for i,sample in enumerate(X_test):
    predictedClass = predictSample(sample, models)
    predictions[i] = predictedClass

correct = 0

for i in range(len(X_test)):
    if predictions[i] == Y_test[i]: correct += 1 

acc = (correct / len(Y_test)) * 100 

print("Perceptron Classfiication")
print(f"Correct Samples Prediction {correct} out of {len(X_test)}")
print(f"Accuracy {acc}")
