import math
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

samples = [[row[0] , row[1]] for row in load_iris().data.tolist()][:10]

def getDistance(vec1,vec2):
    return math.sqrt(sum([(d1-d2)**2 for d1,d2 in zip(vec1,vec2)]))


clusters = [[i] for i in range(len(samples))]
linkage = []
clusterID = list(range(len(samples)))
nextID = len(samples)

while len(clusters) > 1:
    a,b = -1 , -1
    mini = float('inf')

    for i in range(len(clusters)):
        for j in range(i+1 , len(clusters)):
            for p in clusters[i]:
                for q in clusters[j]:
                    dis = getDistance(samples[p],samples[q])
                    if dis < mini:
                        mini = dis
                        a,b = i,j 

    ida , idb = clusterID[a] , clusterID[b]
    linkage.append([ida,idb,mini,len(clusters[a]) + len(clusters[b])])
    clusters[a].extend(clusters[b])
    clusters.pop(b)

    clusterID[a] = nextID
    clusterID.pop(b)
    nextID += 1 


plt.figure(figsize=(10,10))
dendrogram(linkage)
plt.show()





    
