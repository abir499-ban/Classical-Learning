age = [43, 21, 25, 42, 57, 59]
glucose = [99, 65, 79, 75, 87, 61]

def mean(arr):
    s = 0
    for x in arr:
        s += x 

    return s / len(arr)


def Covariance(feat1  , feat2 , n):
    x_mean = mean(feat1)
    y_mean = mean(feat2)

    S = 0

    for x,y in zip(feat1 , feat2):
        S += (x - x_mean) * (y - y_mean)

    return S / n 

if __name__ == "__main__":
    cov = Covariance(age , glucose , len(age))
    print("Covariance: ", cov)
