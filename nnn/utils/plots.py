"""
plots.py: Module with plotting tools. It contains a function to plot the classification boundaries of a 2d classifier, among others.
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
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def plot_boundaries_2d(nn, resolution=100, init=0.0, end=1.0, xlabel = 'w1', ylabel = 'w2', title=None, save_path=None):
    print('Plotting boundaries ...')
    a = np.linspace(init, end, resolution)
    b = np.linspace(init, end, resolution)

    pred_grid = np.zeros(shape=(resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            pred_grid[i, j] = nn.predict([a[i], b[j]])

    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['orange', 'black', 'white'], 256)
    img = plt.imshow(pred_grid, origin='lower', interpolation='quadric', extent=[init, end, init, end], cmap=cmap)
    plt.colorbar(img, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_data(x_train, y_train, x_test, y_test, xlabel = 'A', ylabel = 'B', init=0.0, end=1.0, title=None, save_path=None):
    print('Plotting data ...')
    colors = ['r' if y == 1 else 'b' for y in y_train]
    plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=colors)
    colors = ['r' if y == 1 else 'b' for y in y_test]
    plt.scatter(x_test[:, 0], x_test[:, 1], marker='o', c=colors)
    plt.scatter(x_test[:, 0], x_test[:, 1], marker='*', s=20, c='w')
    plt.xlim(init, end)
    plt.ylim(init, end)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_loss_function_3d(loss_grid, w1, w2, xlabel = 'w1', ylabel = 'w2', title=None, save_path=None):
    print('Plotting loss function ...')

    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.meshgrid(w1, w2)
    ax.plot_surface(x, y, loss_grid, cmap="CMRmap_r", lw=3, linestyles="solid")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
