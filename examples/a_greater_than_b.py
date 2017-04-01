from core.activations import Sigmoid
from core.initializers import GaussianInitializer
from core.neural_network import NeuralNetwork
from core.objectives import CrossEntropy
from utils.plot_boundaries_2d import plot_boundaries_2d
import numpy as np
np.random.seed(2)

data = np.array([[0, 0],
                 [0, 1],
                 [0, 2],
                 [1, 0],
                 [1, 1],
                 [1, 2],
                 [2, 0],
                 [2, 1],
                 [2, 2]])
labels = np.array([int(x[0] > x[1]) for x in data])
train_indices = [0, 1, 3, 4, 5, 7, 8]
test_indices = [2, 6]


nn = NeuralNetwork(input_dim=2, output_dim=1)
nn.add_dense(output_dim=1, activation=Sigmoid(), initializer=GaussianInitializer(gaussian_mean=0.0, gaussian_std=1))
nn.train(x=data[train_indices], y=labels[train_indices], objective=CrossEntropy(), epochs = 200, lr=0.3)


print('Train data')
for i in train_indices:
    f = nn.predict(data[i])
    print('Pred: {0:2f}. Label: {1:2d}'.format(f[0], labels[i]))

print('Test data')
for i in test_indices:
    f = nn.predict(data[i])
    print('Pred: {0:2f}. Label: {1:2d}'.format(f[0], labels[i]))

plot_boundaries_2d(nn, resolution=100, init=0.0, end=2.0, title='Probability that A>B')