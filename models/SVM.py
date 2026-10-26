from sklearn.svm import SVC
import random
from dataset.iris import IrisDataMatrix
from logic.featureSelection.utils import TARGETINDX

def splitData():
    features = []
    target = []

    indices = list(range(len(IrisDataMatrix)))
    random.shuffle(indices)

    for i in indices:
        features.append(IrisDataMatrix[i][:4])
        target.append(TARGETINDX[IrisDataMatrix[i][4]])

    X_train, X_test, Y_train, Y_test = [],[],[],[]

    split = int(0.7 * len(IrisDataMatrix))

    X_train = features[:split]
    Y_train = target[:split]

    X_test = features[split:]
    Y_test = target[split:]

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = splitData()
model = SVC(kernel='linear')
model.fit(X_train , Y_train)

predictions = model.predict(X_test)

accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == Y_test[i]]) /  len(Y_test)

print("SVM CLASSIFICATION (IRIS)")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Classes:", len(set(Y_test)))
print("Correct predictions:",
      sum(a == p for a, p in zip(Y_test, predictions)),
      "out of", len(Y_test))
print("Accuracy (%):", round(accuracy * 100, 2))

    