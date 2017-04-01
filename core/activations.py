"""
activations.py: Activation functions. They require its derivatives
                with respect to their input.
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
