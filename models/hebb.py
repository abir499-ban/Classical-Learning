from . import *


class Hebb(BaseModel):
    def __init__(self, features):
        super().__init__("Hebbian Learning")
        self.w = np.zeros(features)
        self.b = 0
    
    def fit(self, X, y):
        X = encoder(X)
        y = encoder(y)

        for x,t in zip(X,y):
            self.w += x * t
            self.b += t
    

    def predict(self, X):
        X = encoder(X)
        y = bipolar_step(np.dot(X, self.w) + self.b)
        return decoder(y)
    
    

