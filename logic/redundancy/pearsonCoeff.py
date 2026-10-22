import math

# Dataset
age = [43, 21, 25, 42, 57, 59]
glucose = [99, 65, 79, 75, 87, 61]

def SumCol(arr):
    s = 0
    for x in arr:
        s += x

    return s 

def SquareSumCol(arr):
    s = 0
    for x in arr:
        s += x * x

    return s

def SumProductCol(arr1, arr2):
    s = 0
    for x,y in zip(arr1, arr2):
        s += x*y 

    return s

def derivePearsonCoefficient(feat1, feat2):
    n = len(feat1)
    sum_xy = SumProductCol(feat1 , feat2)
    sum_x = SumCol(feat1)
    sum_y = SumCol(feat2)
    sum_xx = SquareSumCol(feat1)
    sum_yy = SquareSumCol(feat2)

    numerator = n * sum_xy - sum_x * sum_y 
    denominator = math.sqrt((n*sum_xx - sum_x ** 2) * (n*sum_yy - sum_y ** 2))

    r = numerator / denominator
    return r




if __name__ == "__main__":
    pearsonCoefficient = derivePearsonCoefficient(age, glucose) 
    print("Pearson Coefficient: ", pearsonCoefficient)
