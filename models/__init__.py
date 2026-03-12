import numpy as np
from activations import step
from utils import encoder, decoder


class BaseModel:
    def __init__(self, *args, **kwargs):
        print(args, kwargs)
        pass 

    def fit(self, X, y):
        pass 

    def prefdict(self, X):
        pass