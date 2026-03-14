import numpy as np

def calculate_MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)