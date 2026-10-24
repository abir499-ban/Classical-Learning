from .utils import FEATURE_NAME, getFeatures , getTarget
from math import sqrt


def getCorelationCoeff(data1, data2):
    n = len(data1)

    sum_xy = sum([x*y for x,y in zip(data1 , data2)])
    sum_x = sum(data1)
    sum_y = sum(data2)
    sum_xx = sum([x*x for x in data1])
    sum_yy = sum([y*y for y in data2])

    numerator = n * sum_xy - sum_x * sum_y 
    denominator = sqrt((n*sum_xx - sum_x**2) * (n*sum_yy - sum_y**2))

    return numerator / denominator


features = getFeatures()
target = getTarget()


featureCorelations = []

for j in range(4):
    r = getCorelationCoeff(features[j] , target)
    featureCorelations.append(r)

RankedFeatures = sorted(range(4) , key=lambda x : featureCorelations[x] , reverse=True)

print("Features Ranked by Pearson Corrleation with Target")

for i in RankedFeatures:
    print(f"Feature : {FEATURE_NAME[i]} | r : {featureCorelations[i]}")

    