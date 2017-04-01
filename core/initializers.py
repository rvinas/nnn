"""
initializers.py: Initializers used for weight initialization
Copyright 2017 Ramon Viñas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np


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
