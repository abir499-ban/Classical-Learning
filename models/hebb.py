import numpy as np
from models import BaseModel


class Hebb(BaseModel):
    def __init__(self, features):
        super().__init__("Hebbian Learning")
        self.w = np.zeros(features)
        self.b = 0
    
    def fit(self, X, y):
        for x,t in zip(X,y):
            self.w += x * t
            self.b += t
    

    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
    

