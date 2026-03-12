import numpy as np


def step(x):
    return np.where(x >= 0, 1, 0)

def bipolar_step(x):
    return np.where(x >= 0, 1, -1)