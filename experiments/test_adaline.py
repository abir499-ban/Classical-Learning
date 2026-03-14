from dataset import load_OR
from models.adaline import Adaline 

def test_adaline():
    X,y = load_OR()

    model = Adaline(features=2, learning_rate=0.0001, epochs=20)
    model.fit(X,y)
    p = model.predict(X)

    print("Input:\n", X)
    print("True Labels:\n", y)
    print("Predictions: \n", p)

    print("")


if __name__ == "__main__":
    test_adaline()