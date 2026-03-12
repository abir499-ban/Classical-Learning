import numpy as np

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

def load_XOR():
    return X, np.array([0,1,1,0])


def load_AND():
    return X, np.array([0,0,0,1])

def load_OR():
    return X , np.array([0,1,1,1])

def load_NAND():
    return X, np.array([1,1,1,0])