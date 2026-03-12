from dataset import load_OR
from models.perceptron import Perceptron




def test_perceptron():
    X,y = load_OR()

    model = Perceptron(features=2, threshold=0.2, learning_rate=0.1, epochs=100)
    model.fit(X,y)
    p = model.predict(X)

    print("Input:\n", X)
    print("True Labels:\n", y)
    print("Predictions: \n", p)

    print("")

if __name__ == "__main__":
    test_perceptron()