import matplotlib.pyplot as plt
import math

IMAGE_FILE_NAME = "train-images-idx3-ubyte"

def read_Image(imageFileName , index):
    with open(imageFileName , 'rb') as f:
        magicNumber = int.from_bytes(f.read(4) , "big")
        imageCount = int.from_bytes(f.read(4) , "big")
        row = int.from_bytes(f.read(4) , "big")
        col = int.from_bytes(f.read(4) , "big")

        imageSize = row * col 
        f.seek(16 + index * imageSize)
        image = []

        for _ in range(row):
            vec = []
            for _ in range(col):
                vec.append(int.from_bytes(f.read(1) , "big"))

            image.append(vec)

    return image 

def transposeMatrix(mat):
    m,n = len(mat) , len(mat[0])
    t = [[-1 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            t[i][j] = mat[j][i]

    return t

def dot(vec1, vec2):
    s = 0

    for x,y in zip(vec1 , vec2):
        s += x*y

    return s 

def mat_Vec(mat , vec):

    res = []

    for row in mat:
        res.append(dot(row , vec))

    return res 

def getColMeans(mat):
    m,n = len(mat) , len(mat[0])
    means = [0 for _ in range(n)]

    for j in range(n):
        for i in range(m):
            means[j] += mat[i][j]

        means[j] /= m 

    return means 

def normalizeVector(v):
    mag = math.sqrt(sum(list(x**2 for x in v)))

    norm = []

    for x in v:
        norm.append(x/mag)

    return norm



def centreImage(image , means):
    m,n = len(image) , len(image[0])
    centeredImage = [[0 for _ in range(n)] for _ in range(m)]

    for j in range(n):
        for i in range(m):
            centeredImage[i][j] = image[i][j] - means[j]

    return centeredImage

def covarianceMatrix(image):
    m,n = len(image) , len(image[0])

    means = getColMeans(image)
    c = centreImage(image , means)

    covMat = [[0 for _ in range(n)] for _ in range(n)]

    cT = transposeMatrix(c)

    for i in range(n):
        for j in range(n):
            for k in range(m):
                covMat[i][j] += cT[i][k] * c[k][j]

            covMat[i][j] /= (m-1)

    return covMat 

def powerIteration(mat , prevBasic):
    n = len(mat)

    v = [1.0] * n 

    for _ in range(40):
        for p in prevBasic:
            prod = dot(p,v)
            for i in range(n):
                v[i] -= prod * p[i]

        v = mat_Vec(mat, v)
        v = normalizeVector(v)

    eigenValue = dot(v , mat_Vec(mat , v))

    return eigenValue , v

def basicPCA(cov , k):
    basics = []
    values = []

    for _ in range(k):
        eigenValues , eigenVector = powerIteration(cov , basics)
        basics.append(eigenVector)
        values.append(eigenValues)

    return values , basics 



def createReducedImage(image , basics):

    reducedImage = []

    for row in image:
        reducedRow = []
        for vec in basics:
            reducedRow.append(dot(row , vec))

        reducedImage.append(reducedRow)

    return reducedImage

def reconstructImage(reducedImage , basics , mean, K):
    reconstructedImage = [[mean[i] for i in range(len(mean))] for _ in range(len(reducedImage))]

    for i in range(len(reducedImage)):
        for j in range(len(mean)):
            for k in range(K):
                reconstructedImage[i][j] += reducedImage[i][k] * basics[k][j]

    return reconstructedImage

    






if __name__=="__main__":
    image = read_Image(IMAGE_FILE_NAME , 5)

    mean , cov = getColMeans(image) , covarianceMatrix(image) 
    centeredImage =  centreImage(image, mean)

    eigenValues , eigenVectors = basicPCA(cov , 14)

    reducedImage = createReducedImage(centeredImage , eigenVectors)
    reconstructedImage = reconstructImage(reducedImage , eigenVectors , mean, 14)


    plt.figure(figsize=(8,4))

    plt.subplot(1,3,1)
    plt.imshow(image , cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(reducedImage , cmap='gray')
    plt.title("Reduced Image")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(reconstructedImage , cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')


    plt.show()





