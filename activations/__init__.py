import numpy as np


def step(x):
    return np.where(x >= 0, 1, 0)

def bipolar_step(x):
    return np.where(x >= 0, 1, -1)

def threshold(y,t_val=0.5):
    return np.where(y > t_val, 1, np.where(y<-t_val, -1, 0))