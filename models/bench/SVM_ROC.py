
from matplotlib import pyplot as plt

from models.bench.prepareDataSet import startifiedSplitDataSet
from sklearn.svm import SVC

def createConfusionMatrix(actual, predicted):
    mat = [[0,0] for _ in range(2)]

    for i in range(len(actual)):
        mat[predicted[i]][actual[i]] += 1

    return mat

def deriveMetrcis(mat):
    TP = mat[1][1]
    FN = mat[0][1]
    FP= mat[1][0]
    TN = mat[0][0]

    accuracy = (TP + TN) / (TP+FP+TN+FN)
    error = 1- accuracy 
    recall = TP / (TP+FN)
    specificity = TN / (TN+FP)
    precision =  TP / (TP + FP)
    FPR = 1 - specificity
    F1 = 2 * precision * recall / (precision + recall)

    ##print All



X_train,Y_train,X_test, Y_test = startifiedSplitDataSet()
model = SVC(kernel="linear")

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

confusionMat = createConfusionMatrix(Y_test , predicted=predictions)

scores = model.decision_function(X_test)

thresholds = sorted(set(scores) , reverse=True)
thresholds = [float('inf')] + thresholds + [float('-inf')]

points = []

for t in thresholds:
    TP,TN,FP,FN=0,0,0,0
    for i in range(len(X_test)):
        predictedClass = 1 if scores[i] >= t else 0
        actualClass = Y_test[i]

        if predictedClass and actualClass: TP += 1
        if predictedClass and not actualClass : FP += 1
        if not predictedClass and actualClass: FN += 1
        if not predictedClass and not actualClass: TN += 1

    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    points.append([FPR,TPR])


points.sort()

plt.figure(figsize=(10,10))
plt.plot(
    [x for x,_ in points],
    [y for _,y in points],
    marker="o",
    label="Setosa vs Rest"
)
plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    label="random"    
)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()








