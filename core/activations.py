import math
import numpy as np


class Activation:
    def __init__(self):
        raise NotImplementedError

    def activation(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        pass

    def activation(self, x):
        return 1 / (1 + math.e ** -x)

    def derivative(self, x):
        s = self.activation(x)
        return s * (1 - s)


class Tanh(Activation):
    def __init__(self):
        pass

    def activation(self, x):
        return np.array([math.tanh(i) for i in x])

    def derivative(self, x):
        return np.array([math.cosh(i) ** -2 for i in x])
