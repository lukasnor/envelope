import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    mpl.use('qtagg')
    x = y = np.arange(0., 10., 0.1)
    X, Y = np.meshgrid(x, y)
    nu = 2*X**3 + 3*X**2*Y - 3*X*Y**2 -2*Y**3
    fig: Figure = plt.figure(dpi=200)
    ax = Axes3D(fig=fig, elev=17, azim=-164)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, nu, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.show()