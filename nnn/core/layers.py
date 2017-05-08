"""
layer.py: Supported layers for the neural network. They must implement both
          the forward and backward pass functions.
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
import numpy as np


class Layer:
    def __init__(self, input_dim, output_dim, activation, initializer):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initializer = initializer


class Dense(Layer):
    def __init__(self, input_dim, output_dim, activation, initializer):
        super(Dense, self).__init__(input_dim, output_dim, activation, initializer)
        self.weights = self.initializer.get_init(
            shape=(output_dim,
                   1 + input_dim))  # Shape of weights: (output_dim, 1 + input_dim,). Note that weights[:, 0] corresponds to bias term for every output unit
        self.der_act_wrt_h = None
        self.input = None

    def forward_pass(self, x):
        """
        Computes f(x)=g(W*x + b), where g is the activation function, W the layer weights and b the bias term.
        In addition, stores the derivative of f with respect to g to be used during the backward pass
        :param x: layer input. Shape: (self.input_dim,)
        :return: f(x)
        """
        assert len(x) == self.input_dim
        self.input = np.concatenate(([1.0], x))  # adds product identity element for bias term
        h = np.dot(self.weights, self.input)
        self.der_act_wrt_h = self.activation.derivative(h)  # this derivate will be used in the backward pass
        return self.activation.activation(h)

    def backward_pass(self, der_loss_wrt_act, lr):
        """
        Computes the loss with respect to every weight in the layer according to backpropagation algorithm.
        Updates all the weights in layer according to Stochastic Gradient Descent. Returns the backward loss to be transmitted to the previous layer.
        :param lr: learning rate
        :param der_loss_wrt_act: derivative of the loss with respect to the activation function of dense's output. Shape: (self.output_dim,)
        :return: backward loss. Shape: (self.input_dim,)
        """
        assert len(der_loss_wrt_act) == self.output_dim
        der_h_wrt_w = np.repeat(np.expand_dims(self.input, axis=0), self.output_dim, axis=0)
        aux = der_loss_wrt_act * self.der_act_wrt_h
        der_loss_wrt_w = np.expand_dims(aux, axis=-1) * der_h_wrt_w
        backward = np.dot(aux, self.weights[:, 1:])  # computes the loss to be transmitted to the previous layer (note that bias term is being excluded)
        self.weights -= lr * der_loss_wrt_w
        return np.squeeze(backward)

