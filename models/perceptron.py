from . import *;

class Perceptron(BaseModel):
    def __init__(self, features, threshold, learning_rate,epochs):
        super().__init__("Perceptron Learning")
        self.w = np.zeros(features)
        self.b = 0 
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        X,y = encoder(X), encoder(y)
        
        for ep in range(self.epochs):
            error = 0
            for x,t in zip(X,y):
                y_in = np.dot(self.w,x) + self.b
                y_out = threshold(y_in, self.threshold)
                
                if y_out != t:
                    self.w += self.learning_rate * x * t
                    self.b += self.learning_rate * t
                    error += 1
            
            if error == 0:
                print("Training converged")
                break 
            
            print(f"Epoch : {ep + 1}, errors : {error}")
    

    def predict(self, X):
        X = encoder(X)
        y_in = np.dot(X,self.w) + self.b
        y_out = threshold(y_in, self.threshold)
        return decoder(y_out)
            

