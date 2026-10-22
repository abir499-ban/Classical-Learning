from re import L


age = [43, 21, 25, 42, 57, 59]
glucose = [99, 65, 79, 75, 87, 61]

contigencyTable = [[0 for _ in range(2)] for _ in range(2)]
## Table
## Glucose :     Low High
## Age
## Low
## Low

AGE_THRESHOLD = 42.5
GLUCOSE_THRESHOLD = 77

for x,y in zip(age , glucose):
    ageIndx = 0 if x < AGE_THRESHOLD else 1
    glucoseIndx = 0 if x < GLUCOSE_THRESHOLD else 1

    contigencyTable[ageIndx][glucoseIndx] += 1


rowSums = [0 for _ in range(2)]
colSums = [0 for _ in range(2)]

for i in range(2):
    for j in range(2):
        rowSums[i] += contigencyTable[i][j]
        colSums[j] += contigencyTable[i][j]

expectedTable = [[0 for _ in range(2)] for _ in range(2)]
total = sum(rowSums)

for i in range(2):
    for j in range(2):
        expectedTable[i][j] = (rowSums[i] * colSums[j]) / total  


chiSq = 0

for i in range(2):
    for j in range(2):
        numerator = (contigencyTable[i][j] - expectedTable[i][j]) ** 2
        chiSq += numerator / expectedTable[i][j]

print("Chi-square statistic:", chiSq)