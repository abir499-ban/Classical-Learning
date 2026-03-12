import numpy as np


def encoder(x):
    return np.where(x == 0 , -1, 1)

def decoder(x):
    return np.where(x == -1, 0, 1)