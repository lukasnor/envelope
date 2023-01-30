import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes._base import _AxesBase
from matplotlib.figure import Figure

if __name__ == "__main__":
    mpl.use('qtagg')
    x = y = np.arange(0., 10., 0.1)
    X, Y = np.meshgrid(x, y)
    eta = X**2 + Y**2 + X*Y + 3*(X+Y)
    nu = 2*X**3 + 3*X**2*Y - 3*X*Y**2 -2*Y**3 + 9* X**2 - 9 * Y**2 + 9 *X - 9 * Y
    fig, (ax1, ax2) = plt.subplots(ncols=2
                            , dpi=300
                            , tight_layout=True
                            , subplot_kw=dict(projection="3d", elev=17, azim=-164, xlabel="q", ylabel="p"))
    fig.suptitle("Actions of the Casimirs on an irrep H_p,q", )
    assert isinstance(ax1, _AxesBase)
    ax1.plot_surface(X, Y, eta, cmap=cm.coolwarm, antialiased=False)
    ax1.set_title("Second order")
    ax2.plot_surface(X, Y, nu, cmap=cm.coolwarm, antialiased=False)
    ax2.set_title("Third order")
    #plt.show()
    plt.close(fig)

    x = y = np.arange(-20., 20., 0.1)
    X, Y = np.meshgrid(x, y)
    eta = X**2 + Y**2 + X*Y + 3*(X+Y)
    fig: Figure = plt.figure(dpi=300)
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, eta)
    plt.show()