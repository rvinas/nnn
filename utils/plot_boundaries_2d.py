import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_boundaries_2d(nn, resolution=100, init=0.0, end=2.0, title=None, save_path=None):
    a = np.linspace(init, end, resolution)
    b = np.linspace(init, end, resolution)

    pred_grid = np.zeros(shape=(resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            pred_grid[i, j] = nn.predict([a[i], b[j]])

    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['orange', 'black', 'white'], 256)
    img = plt.imshow(pred_grid, origin='lower', interpolation='quadric', extent=[init, end, init, end], cmap=cmap)
    plt.colorbar(img, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)
