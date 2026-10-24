


from .utils import FEATURE_NAME, getFeatures


def getVariance(data):
    mean = sum(data) /  len(data)

    squared_diffs = [(x-mean) **2 for x in data]

    return sum(squared_diffs) /  len(data)

features = getFeatures()

variance = []

for f in range(4):
    variance.append(getVariance(features[f]))

sortedVariance = sorted(range(4) , key=lambda x : variance[x] , reverse=True)

print("variance Ranking")

for r in sortedVariance:
    print(f"Feature {FEATURE_NAME[r]} : {variance[r]}")

