import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes._base import _AxesBase
from numpy import sqrt
from numpy import arctan as atan

if __name__ == "__main__":
    mpl.use('qtagg')
    # q, p in [(8, 0), (5, 4), (4, 5), (0, 8)]
    x = y = np.arange(0., 8.1, 0.1)
    X, Y = np.meshgrid(x, y)
    eta = X ** 2 + Y ** 2 + X * Y + 3 * (X + Y)
    nu = 2 * X ** 3 + 3 * X ** 2 * Y - 3 * X * Y ** 2 - 2 * Y ** 3 + 9 * X ** 2 - 9 * Y ** 2 + 9 * X - 9 * Y
    fig, (ax1, ax2) = plt.subplots(ncols=2
                                   , dpi=300
                                   , tight_layout=True
                                   , subplot_kw=dict(projection="3d", elev=17, azim=-164, xlabel="q", ylabel="p"))
    fig.suptitle("Actions of the Casimirs on an irrep H_p,q", )
    assert isinstance(ax1, _AxesBase)
    ax1.set_title("Second order")
    start = 2*atan(-4*sqrt(3)/15 + sqrt(273)/15)
    end = 2*atan(4*sqrt(3)/15 + sqrt(273)/15)
    t = np.arange(start, end, 0.01)
    a = 88
    x = 2 * np.sqrt(a + 3) * np.cos(t)
    y = 2 * np.sqrt(a/3 + 1) * np.sin(t)
    q = (x+y-2)/2
    p = (y-x-2)/2
    ax1.plot(q, p, zs=a, color="orange")
    ax1.plot(q, p, zs=0, color="orange", linestyle="dashed")
    ax1.plot_wireframe(X, Y, eta, cmap=cm.coolwarm, rcount=10, ccount=10)
    ax2.plot(q, p, 2 * q ** 3 + 3 * q ** 2 * p - 3 * q * p ** 2 - 2 * p ** 3 + 9 * q ** 2 - 9 * p ** 2 + 9 * q - 9 * p,
             color="orange")

    ax2.plot_wireframe(X, Y, nu, cmap=cm.coolwarm, rcount=10, ccount=10)
    assert isinstance(ax2, _AxesBase)
    ax2.plot(q, p, zs=ax2.zz_dataLim.bounds[0], color="orange", linestyle="dashed")
    ax2.set_title("Third order")
    plt.show()
    #plt.close()

    # x = y = np.arange(-20., 20., 0.1)
    # X, Y = np.meshgrid(x, y)
    # eta = X ** 2 + Y ** 2 + X * Y + 3 * (X + Y)
    # fig: Figure = plt.figure(dpi=300)
    # ax = fig.add_subplot(projection="3d")
    # ax.plot_surface(X, Y, eta)
    # plt.show()
    #plt.close()

    # start = -2*atan(-6*sqrt(5)/11 - sqrt(15)/11 + 2/11 + 4*sqrt(3)/11)
    # end = -2*atan(-6*sqrt(5)/11 - 4*sqrt(3)/11 + 2/11 + sqrt(15)/11)
    # t = np.arange(start, end, 0.01)
    # a = 9
    # x = lambda t: 2 * np.sqrt(a + 3) * np.cos(t)
    # y = lambda t: np.sqrt(4 / 3 * (a + 3)) * np.sin(t)
    # q = (lambda x, y: lambda t: (x(t) + y(t) - 2) / 2)(x, y)
    # p = (lambda x, y: lambda t: (y(t) - x(t) - 2) / 2)(x, y)
    # plt.plot(q(t), p(t))
    # plt.scatter(q(t[0:50:10]), p(t[0:50:10]), color='r')
    # plt.hlines(0, 0, 3, color="black")
    # plt.vlines(0, 0, 3, color="black")
    # plt.scatter([0, 2, 1], [2, 0, 1], color="grey")
    # plt.plot([0, 1], [0, 1])
    # plt.scatter([q(np.pi/2)], [p(np.pi/2)], color="purple")
    # plt.show()
    #plt.close()