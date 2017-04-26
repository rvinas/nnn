"""
a_greater_than_b.py: Example demonstrating that a linearly separable dataset can
                     be classified using a neural network without any hidden layer.
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
from nnn.utils.plots import plot_boundaries_2d, plot_data, plot_loss_function_3d
from nnn.utils.loss_grid import loss_grid_nn
import numpy as np

np.random.seed(15)

n_samples_train = 100
n_samples_test = 25
x_train = np.random.rand(n_samples_train, 2)
x_test = np.random.rand(n_samples_test, 2)
y_train = np.array([int(x[0] > x[1]) for x in x_train])
y_test = np.array([int(x[0] > x[1]) for x in x_test])

plot_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, init=0, end=1, title='Data plot. Star points are test samples', save_path='a_greater_than_b_data.png')

nn = NeuralNetwork(input_dim=2, output_dim=1)
nn.add_dense(output_dim=1, activation=Sigmoid(), initializer=GaussianInitializer(gaussian_mean=0.0, gaussian_std=0.3))
nn.train(x=x_train, y=y_train, objective=CrossEntropy(), epochs=25, lr=0.3)

print('Train data')
for i in range(n_samples_train):
    f = nn.predict(x_train[i])
    print('Pred: {0:2f}. Label: {1:2d}'.format(f[0], y_train[i]))

print('Test data')
for i in range(n_samples_test):
    f = nn.predict(x_test[i])
    print('Pred: {0:2f}. Label: {1:2d}'.format(f[0], y_test[i]))

plot_boundaries_2d(nn, resolution=100, init=0.0, end=2.0, title='Probability that A>B', save_path='a_greater_than_b_boundaries.png')

loss_g, w1, w2 = loss_grid_nn(nn=nn, objective=CrossEntropy(), x_train=x_train, y_train=y_train, resolution=20,
                           w1_init=-5, w1_end=5, w2_init=-5, w2_end=5)
plot_loss_function_3d(loss_grid=loss_g, w1=w1, w2=w2, title='Loss function', save_path=None)
