"""
a_equal_to_b.py: Example demonstrating that a non linearly separable dataset
                 requires at least a hidden layer in order to perform a classification.
Copyright 2017 Ramon Vinas

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
from nnn.core.activations import Sigmoid
from nnn.core.initializers import GaussianInitializer
from nnn.core.neural_network import NeuralNetwork
from nnn.core.objectives import CrossEntropy
from nnn.utils.plots import plot_boundaries_2d, plot_data
import numpy as np
np.random.seed(15) # Note that the neural network is highly sensitive to initial conditions.

n_samples_train = 100
n_samples_test = 25
x_train = np.random.rand(n_samples_train, 2)
x_test = np.random.rand(n_samples_test, 2)
y_train = np.array([int(abs(x[0]-x[1]) > 0.2) for x in x_train])
y_test = np.array([int(abs(x[0]-x[1]) > 0.2) for x in x_test])

plot_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, init=0, end=1, title='Data plot. Star points are test samples', save_path='a_aprox_b_data.png')

nn = NeuralNetwork(input_dim=2, output_dim=1)
# Try commenting the next line: the neural network won't be able to classify the samples, because data isn't linearly separable. Thus, we need a hidden layer.
nn.add_dense(output_dim=2, activation=Sigmoid(), initializer=GaussianInitializer(gaussian_mean=0.0, gaussian_std=1))
nn.add_dense(output_dim=1, activation=Sigmoid(), initializer=GaussianInitializer(gaussian_mean=0.0, gaussian_std=1))
nn.train(x=x_train, y=y_train, objective=CrossEntropy(), epochs=25, lr=0.3)

print('Train data')
for i in range(n_samples_train):
    f = nn.predict(x_train[i])
    print('Pred: {0:2f}. Label: {1:2d}'.format(f[0], y_train[i]))

print('Test data')
for i in range(n_samples_test):
    f = nn.predict(x_test[i])
    print('Pred: {0:2f}. Label: {1:2d}'.format(f[0], y_test[i]))

plot_boundaries_2d(nn, resolution=100, init=0.0, end=2.0, title='Probability that |A-B|>0.2', save_path='a_aprox_b_boundaries.png')
