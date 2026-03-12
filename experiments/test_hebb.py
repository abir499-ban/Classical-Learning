from dataset import load_XOR, load_OR
from models.hebb import Hebb

def test_hebb():
    X,y = load_OR()
    model = Hebb(features=2)

    model.fit(X,y)

    p = model.predict(X)

    print("Input:\n", X)
    print("True Labels:\n", y)
    print("Predictions: \n", p)

    print("")


if __name__ == "__main__":
    test_hebb()
