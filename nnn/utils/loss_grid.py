"""
loss_grid.py: Computation of the loss grid for a given neural network and objective function
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


def loss_grid_nn(nn, objective, x_train, y_train, resolution=100, w1_init=0.0, w1_end=2.0, w2_init=0.0, w2_end=2.0):
    w1 = np.linspace(w1_init, w1_end, resolution)
    w2 = np.linspace(w2_init, w2_end, resolution)

    loss_g = np.zeros(shape=(resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            nn.layers[0].weights = np.array([[1, w1[i], w2[j]]])
            for k in range(len(x_train)):
                pred = nn.predict(x_train[k])
                loss_g[i, j] += objective.get_loss(y_train[k], pred)

    return loss_g, w1, w2