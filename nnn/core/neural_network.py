"""
neural_network.py: Simple neural network from scratch
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
from nnn.core.activations import *
from nnn.core.layers import *
from nnn.core.initializers import *


class NeuralNetwork:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.last_output_dim = None
        self.layers = []

    def add_dense(self, output_dim, activation=Sigmoid,
                  initializer=GaussianInitializer(gaussian_mean=0.0, gaussian_std=0.1)):
        """
        Adds a dense (fully connected) layer to the network.
        :param output_dim: number of output units
        :param activation: activation function. See core/activations
        :param initializer: weight initializer. See core/initializers
        """
        input_dim = self.last_output_dim
        if input_dim is None:
            input_dim = self.input_dim
        layer = Dense(input_dim=input_dim, output_dim=output_dim, activation=activation, initializer=initializer)
        self.last_output_dim = output_dim
        self.layers += [layer]

    def _forward_pass(self, input):
        """
        Computes the forward step through all the layers in the network.
        :param input: network's input. Shape: (self.input_dim,)
        :return: network's output. Shape: (self.output_dim,)
        """
        assert len(input) == self.input_dim
        h = input
        for layer in self.layers:
            h = layer.forward_pass(h)
        return h

    def _backward_pass(self, der_loss_wrt_act, lr):
        """
        Computes the backward step through all the layers in the network, updating all weights.
        :param der_loss_wrt_act: derivative of the loss function with respect to the output activation. Shape: (self.output_dim,)
        :param lr: learning rate
        """
        assert len(der_loss_wrt_act) == self.output_dim
        der_out_wrt_act = np.array(der_loss_wrt_act)
        for layer in reversed(self.layers):
            der_out_wrt_act = layer.backward_pass(der_out_wrt_act, lr)

    def train(self, x, y, objective, epochs=5, lr=0.01):
        """
        Trains the network using batches of size 1. TODO: batch support
        :param x: Input data. Shape: (?, self.input_dim,)
        :param y: Input labels. Shape: (?, self.output_dim,)
        :param objective: Objective function. See core/objectives
        :param epochs: Number of training epochs
        :param lr: learning rate
        """
        assert self.last_output_dim == self.output_dim
        indices = list(range(len(x)))

        for epoch in range(epochs):
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            print('\nEpoch: ', epoch)
            for i in range(len(x)):
                output = self._forward_pass(x[i])
                loss = objective.get_loss(y[i], output)
                self._backward_pass(objective.derivative(y[i], output), lr)
                print('iter: {0:2d}, loss: {1:0.2f}'.format(i, loss))

    def predict(self, x):
        """
        Computes the network forward step.
        :param x: Input data. Shape: (self.input_dim,)
        :return: prediction. Shape: (self.output_dim,)
        """
        return self._forward_pass(x)