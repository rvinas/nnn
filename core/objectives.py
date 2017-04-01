import math


class Objective:
    def __init__(self):
        raise NotImplementedError

    def get_loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class CrossEntropy(Objective):
    def __init__(self):
        pass

    def get_loss(self, y_true, y_pred):
        return -y_true * math.log(y_pred) - (1 - y_true) * math.log(1 - y_pred)

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
