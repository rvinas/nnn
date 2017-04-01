import numpy as np
np.random.seed(0)


class Initializer:
    def __init__(self):
        raise NotImplementedError

    def get_init(self, shape):
        raise NotImplementedError


class GaussianInitializer:
    def __init__(self, gaussian_mean=0.0, gaussian_std=0.1):
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std

    def get_init(self, shape):
        return np.random.normal(self.gaussian_mean, self.gaussian_std, size=shape)
