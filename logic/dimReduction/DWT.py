import matplotlib.pyplot as plt

IMAGE_FILE_NAME = "train-images-idx3-ubyte"

def readImage(imageFilePath, index):
    with open(file=imageFilePath , mode='rb') as f:
        _ = int.from_bytes(f.read(4) , "big")
        _ = int.from_bytes(f.read(4) , "big")
        rows = int.from_bytes(f.read(4) , "big")
        cols = int.from_bytes(f.read(4) , "big")

        imageSize = rows * cols 
        images = []
        f.seek(16 + index*imageSize)

        for _ in range(rows):
            v = []
            for _ in range(cols):
                v.append(int.from_bytes(f.read(1) , 'big'))

            images.append(v)

        return images 

def haar_DWT(image):
    m,n = len(image) , len(image[0])

    lowHorizontal , highHorizontal = [] , []

    for i in range(m):
        lowVec, highVec = [] , []

        for j in range(0, n , 2):
            a,b = image[i][j] , image[i][j+1]
            lowVec.append((a+b) / 2)
            highVec.append((a-b) / 2)

        lowHorizontal.append(lowVec)
        highHorizontal.append(highVec)

    LL,LH,HL,HH = [],[],[],[]

    R,C = len(lowHorizontal) , len(lowHorizontal[0])

    for j in range(C):
        vec1, vec2 = [],[]

        for i in range(0,R,2):
            a,b = lowHorizontal[i][j] , lowHorizontal[i+1][j]
            vec1.append((a+b) / 2)
            vec2.append((a-b) / 2)

        LL.append(vec1)
        LH.append(vec2)

    for j in range(C):
            vec1, vec2 = [],[]
    
            for i in range(0,R,2):
                a,b = highHorizontal[i][j] , highHorizontal[i+1][j]
                vec1.append((a+b) / 2)
                vec2.append((a-b) / 2)
    
            HL.append(vec1)
            HH.append(vec2)

    return LL,LH,HL,HH 


if __name__ == "__main__":
    image = readImage(IMAGE_FILE_NAME, 9)
    LL,LH,HL,HH = haar_DWT(image)

    plt.figure(figsize=(10,10))

    plt.subplot(2,3,1)
    plt.imshow(image, cmap="gray")
    plt.title("Original IMage")
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(LL, cmap="gray")
    plt.title("Low Horizontal + Low Vertical")
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(LH, cmap="gray")
    plt.title("Low Horizontal + High Vertical")
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.imshow(HL, cmap="gray")
    plt.title("High Horizontal + Low Vertical")
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(HH, cmap="gray")
    plt.title("High Horizontal + High Vertical")
    plt.axis('off')

    plt.show()


    
    

        
        