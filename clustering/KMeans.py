from copy import deepcopy
import matplotlib.pyplot as plt
from dataset.iris import IrisDataMatrix
import math

def Eucleaden(v1,v2):
    S = sum([(d1-d2) ** 2 for d1,d2 in zip(v1,v2)])

    return math.sqrt(S)

def findCentroids(features, K):
    c = [[] for _ in range(K)]

    for i in range(K):
        c[i] = deepcopy(features[i])

    return c 

def clusterSamples(features, centroids):
    n = len(features)
    clusterLabels = [-1] * n
    clusters = [[] for _ in range(len(centroids))]

    for i,x in enumerate(features):
        mini,cluster = float('inf') , -1
        for label,c in enumerate(centroids):
            dis = Eucleaden(c,x)
            if dis < mini:
                mini = dis 
                cluster = label

        clusters[cluster].append(x)
        clusterLabels[i] = cluster 

    return clusterLabels , clusters


def runKMeans(features , K):

    centroids = findCentroids(features, K)
    
    while True:
        clusterLabels , clusters = clusterSamples(features , centroids)
        new_Centroids = []

        for cluster in clusters:
            nc = [0] * len(features[0])
            for j in range(len(features[0])):
                for point in cluster:
                    nc[j] += point[j]


            nc = [x/len(cluster) for x in nc]
            new_Centroids.append(nc)

        same = True 

        for i in range(K):
            if centroids[i] != new_Centroids[i]:
                same = False 
                break 

        if same: break 


        centroids = new_Centroids

    return clusterLabels , clusters 

def calculateA(clusterPoints, point):

    disSum = 0
    for p in clusterPoints:
        if p == point: continue 
        disSum += Eucleaden(p,point)

    return disSum / (len(clusterPoints)-1)

def calculateB(clusters, point , clusterIndex):

    minAvg = float('inf')
    for idx, clusterPoints in enumerate(clusters):
        if idx == clusterIndex : continue
        avg = calculateA(clusterPoints , point)
        minAvg = min(minAvg , avg)

    return minAvg

def returnSilhoutteIndex(features , clusterLabels, clusters):
    si = 0 

    for i,point in enumerate(features):
        clusterIndex = clusterLabels[i]
        a,b = calculateA(clusters[clusterIndex] , point) , calculateB(clusters, point, clusterIndex)
        si += (b-a) / max(a,b)

    return si / len(features)





features = []

for r in IrisDataMatrix:
    features.append(r[:4])

k_values = []
si_values = [] 


for K in range(2,5):
    labels, clusters = runKMeans(features ,K)
    SIndex = returnSilhoutteIndex(features, labels, clusters)
    k_values.append(K)
    si_values.append(SIndex)


plt.figure(figsize=(10,10))

plt.plot(
    k_values,
    si_values,
    marker='o'
)

print("SilHoute Index Values")

for i,si in enumerate(si_values):
    print(f"K value : {i+2} | SIlhoutte Index Value : {si}")

plt.xlabel("K Values")
plt.ylabel("Silhoutte index")

plt.title("K vs Silhoutte for KMeans Clustering algo")

plt.show()
    

        