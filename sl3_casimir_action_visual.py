import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes._base import _AxesBase
import matplotlib.colors as colors
from numpy import sqrt
from numpy import arctan as atan


def casimir2(n, q, p):
    return q ** 2 + 2 * q * p / n + p ** 2 + (n + 1) * (q + p)


def casimir3(n, q, p):
    return q ** 3 + 3 * q ** 2 * p / n - 3 * q * p ** 2 / n - p ** 3 \
        + 3 * (n + 1) * (q ** 2 - p ** 2) / 2 \
        + (n + 1) ** 2 * (q - p) / 2

def plot_sl3_casimir():
    mpl.use('qtagg')
    # q, p in [(8, 0), (5, 4), (4, 5), (0, 8)]
    x = y = np.arange(0., 8.1, 0.1)
    X, Y = np.meshgrid(x, y)
    N = 2  # The meshgrid works for all N > 1, but the contour line formulas do not work
    eta = casimir2(N, X, Y)
    nu = casimir3(N, X, Y)
    fig, (ax1, ax2) = plt.subplots(ncols=2
                                   , dpi=300
                                   , tight_layout=True
                                   , subplot_kw=dict(projection="3d", elev=17, azim=-164, xlabel="q", ylabel="p"))
    fig.suptitle("Actions of the Casimirs on an irrep H_p,q", )
    assert isinstance(ax1, _AxesBase)
    ax1.set_title("Second order")
    start = 2 * atan(-4 * sqrt(3) / 15 + sqrt(273) / 15)
    end = 2 * atan(4 * sqrt(3) / 15 + sqrt(273) / 15)
    t = np.arange(start, end, 0.01)
    a = 88
    x = 2 * np.sqrt(a + 3) * np.cos(t)
    y = 2 * np.sqrt(a / 3 + 1) * np.sin(t)
    q = (x + y - N) / 2
    p = (y - x - N) / 2
    ax1.plot(q, p, zs=a, color="orange")
    ax1.plot(q, p, zs=0, color="orange", linestyle="dashed")
    ax1.plot_wireframe(X, Y, eta, cmap=cm.coolwarm, rcount=10, ccount=10)
    ax2.plot(q, p, casimir3(N, q, p), color="orange")

    ax2.plot_wireframe(X, Y, nu, cmap=cm.coolwarm, rcount=10, ccount=10)
    assert isinstance(ax2, _AxesBase)
    ax2.plot(q, p, zs=ax2.zz_dataLim.bounds[0], color="orange", linestyle="dashed")
    ax2.set_title("Third order")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    # plot_sl3_casimir()
    x_min = -2.
    x_max = 10.
    x = y = np.arange(x_min, x_max, 0.1)
    X, Y = np.meshgrid(x, y)
    for N in range(2, 3):
        plt.figure(figsize=(16, 16))
        Z = casimir3(N, X, Y)
        plt.contourf(X, Y, Z, levels=100)
        plt.contour(X, Y, casimir2(N, X, Y))
        plt.hlines([0.0], x_min, x_max)
        plt.vlines([0.0], x_min, x_max)
        plt.title("N = "+str(N))
        plt.show()