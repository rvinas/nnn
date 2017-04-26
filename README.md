# Numpy Neural Network
Neural Network from scratch in Python exclusively using Numpy.
## Overview

This project consists of a neural network implementation from scratch. Modules are organized in a way that intends to provide both an understandable implementation of neural networks and a user-friendly API.
The project is structured as follows:
- `nnn`
    - `core/`: Network main functionalities. 
        - `neural_network.py`: Implementation of the neural network, including functionalities such as building a custom network and training it with backpropagation + stochastic gradient descent.
        - `layers.py`: Defines layers that can take part in the neural network by describing its behavior at the forward and backward steps.
        - `initializers.py`: Functions used to initialize network's weights.
        - `activations.py`: Activation functions that may be used to add nonlinearities to the model.
        - `objectives.py`: Functions to be optimized by adjusting network's parameters.
    - `utils/`: Other utilities.
        - `plots.py`: Module with plotting tools. It contains a function to plot the classification boundaries of a 2d classifier, among others.
        - `loss_grid.py`: Computation of the loss grid for a given neural network and objective function
    - `examples`: Simple instructive examples. See [examples/README.md](examples/)
        - `a_greater_than_b.py`: Example demonstrating that a linearly separable dataset can be classified using a neural network without any hidden layer.
        - `a_aprox_b.py`: Example demonstrating that a non linearly separable dataset requires at least a hidden layer in order to classify samples correctly.

## Prerequisites

- Python 3.5
- Pip 9.0.1

Note: Not tested with other python versions.

## Installation
Once you have met the prerequisites, a single step is required to install this software:
1. Run `sudo pip3 install -r requirements.txt`

This will install `numpy` (the only required external library to run the neural network) and `matplotlib` (only needed to plot classifier boundaries when running an example).

## Further improvements

There are several functionalities that may be implemented to make this software more useful:
- Other types of layers: LSTM, CNN, embeddings,...
- Batches
- More optimizers other than Stochastic Gradient Descent
- More activations
- More initializers
- More objective functions
- Regularization
- Parallelization
- Automatic differentiation
