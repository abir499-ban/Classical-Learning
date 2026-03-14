from . import *;


class Adaline(BaseModel):
    def __init__(self, features, learning_rate=0.1, epochs=100):
        super().__init__("Adaline learning model")
        self.w = np.zeros(features)
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        X,y = encoder(X), encoder(y)
        for e in range(self.epochs):
            mse = 0
            for x,t in zip(X,y):
                y_in = np.dot(self.w, x) + self.b 
                error = t-y_in
                
                self.w += self.learning_rate * error * x 
                self.b += self.learning_rate * error
                mse += error ** 2
            
            mse /=  len(X)
            print(f"Epoch : {e+1}, Mean Square Error : {mse}")

            if mse <=  0.001:
                print(f"Model training converged. Training complete after {e} epochs")
                break 
            
    
    def predict(self, X):
        X = encoder(X) 
        y_in = np.dot(X, self.w) + self.b 
        return decoder(bipolar_step(y_in))