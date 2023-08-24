import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sympy import symbols, IndexedBase, Idx, Sum, Indexed, Wild, factor, Function, groebner
import matplotlib as mpl


def plot_sl4_casimir_actions():
    mpl.use('qtagg')
    # q, p in [(8, 0), (5, 4), (4, 5), (0, 8)]
    x = y = np.arange(0, 10, 0.1)
    X, Y = np.meshgrid(x, y)
    eta = 3 * X ** 2 + 3 * Y ** 2 + 2 * X * Y + 12 * (X + Y)
    nu = X ** 3 + X ** 2 * Y - X * Y ** 2 - Y ** 3 + 6 * X ** 2 - 6 * Y ** 2 + 8 * X - 8 * Y
    fig, (ax1, ax2) = plt.subplots(ncols=2
                                   , dpi=300
                                   , tight_layout=True
                                   , subplot_kw=dict(projection="3d", elev=17, azim=-164, xlabel="q", ylabel="p"))
    fig.suptitle("Actions of the Casimirs on an irrep H_p,q", )
    from matplotlib.axes._base import _AxesBase
    assert isinstance(ax1, _AxesBase)
    ax1.set_title("Second order")
    # start = 2 * np.atan(-4 * np.sqrt(3) / 15 + np.sqrt(273) / 15)
    # end = 2 * np.atan(4 * np.sqrt(3) / 15 + np.sqrt(273) / 15)
    # t = np.arange(start, end, 0.01)
    # a = 88
    # x = 2 * np.sqrt(a + 3) * np.cos(t)
    # y = 2 * np.sqrt(a / 3 + 1) * np.sin(t)
    # q = (x + y - 2) / 2
    # p = (y - x - 2) / 2
    # ax1.plot(q, p, zs=a, color="orange")
    # ax1.plot(q, p, zs=0, color="orange", linestyle="dashed")
    ax1.plot_surface(X, Y, eta, cmap=cm.coolwarm)  # , rcount=10, ccount=10)
    # ax2.plot(q, p, 2 * q ** 3 + 3 * q ** 2 * p - 3 * q * p ** 2 - 2 * p ** 3 + 9 * q ** 2 - 9 * p ** 2 + 9 * q - 9 * p,
    #          color="orange")
    ax2.plot_surface(X, Y, nu, cmap=cm.coolwarm)  # , rcount=10, ccount=10)
    assert isinstance(ax2, _AxesBase)
    # ax2.plot(q, p, zs=ax2.zz_dataLim.bounds[0], color="orange", linestyle="dashed")
    ax2.set_title("Third order")
    plt.show()
    # plt.close()


def calc_casimir_actions():
    h1, hn, n = symbols("h_1 h_n n")
    z1 = n * h1 + hn
    zn = -h1 + hn
    znplus1 = -z1 - (n - 1) * zn

    psi1 = z1 * znplus1 + z1 * (n - 1) * zn + znplus1 * (n - 1) * zn + (n - 1) * (n - 2) / 2 * zn ** 2
    psi1.expand()
    psi1.expand().simplify()
    psi1.expand().as_poly(h1, hn)
    psi1.expand().as_poly(h1, hn).replace(n, 3)

    psi1.replace(n, 3).expand().as_poly(h1, hn)

    psi1.replace(n, 2).expand().as_poly(h1, hn)

    psi1.replace(n, 3).expand().as_poly(h1, hn)


if __name__ == "__main__":

    # These are "handcalculated" values for the different casimir actions
    # They are meant as a comparative guide to check against for the general actions

    # TODO n = 2
    h1, h2, h3, h4 = symbols("h1 h2 h3 h4")
    x1 = 2 * h1 + h2
    x2 = -h1 + h2
    x3 = - x1 - x2

    sl3psi1 = x1*x2 + x1*x3 + x2*x3
    sl3phi1 = sl3psi1.subs(h1, h1+1).subs(h2, h2+1).expand()
    sl3psi2 = x1*x2*x3
    sl3phi2 = sl3psi2.subs(h1, h1+1).subs(h2, h2+1).expand()

    # n = 3
    y1 = 3 * h1 + 2 * h2 + h3
    y2 = -h1 + 2 * h2 + h3
    y3 = - h1 - 2 * h2 + h3
    y4 = - y1 - y2 - y3

    sl4psi1 = y1 * y2 + y1 * y3 + y1 * y4 + y2 * y3 + y2 * y4 + y3 * y4
    sl4phi1 = sl4psi1.subs(h1, h1+1).subs(h2, 1).subs(h3, h3+1).expand()/(-2)
    sl4psi2 = y1 * y4 * (y2 + y3) + (y1 + y4)*(y2*y3)
    sl4phi2 = sl4psi2.subs(h1, h1+1).subs(h2, 1).subs(h3, h3+1).expand()/(8)
    sl4psi3 = y1 * y2 * y3 * y4
    sl4phi3 = sl4psi3.subs(h1, h1+1).subs(h2, 1).subs(h3, h3+1).expand()

    print("sl4phi1", sl4phi1)
    print("sl4phi2", sl4phi2)
    print("NEW: sl4phi3", sl4phi3)


    # n = 4
    z1 = 4 * h1 + 3 * h2 + 2 * h3 + h4
    z2 = - h1 + 3 * h2 + 2 * h3 + h4
    z3 = - h1 - 2 * h2 + 2 * h3 + h4
    z4 = - h1 - 2 * h2 - 3 * h3 + h4
    z5 = - z1 - z2 - z3 - z4

    sl5psi1 = z1 * z5 + (z1 + z5) * (z2 + z3 + z4) + (z2 * z3 + z2 * z4 + z3 * z4)
    sl5phi1 = sl5psi1.subs(h1, h1 + 1).subs(h2, 1).subs(h3, 1).subs(h4, h4 + 1).expand() / (-5)
    sl5psi2 = z1 * z5 * (z2 + z3 + z4) + (z1 + z5) * (z2 * z3 + z2 * z4 + z3 * z4) + (z2 * z3 * z4)
    sl5phi2 = sl5psi2.subs(h1, h1 + 1).subs(h2, 1).subs(h3, 1).subs(h4, h4 + 1).expand() / (5)
    sl5psi3 = z1 * z5 * (z2 * z3 + z2 * z4 + z3 * z4) + (z1+z5)*(z2*z3*z4)
    sl5phi3 = sl5psi3.subs(h1, h1 + 1).subs(h2, 1).subs(h3, 1).subs(h4, h4 + 1).expand()
    sl5psi4 = z1*z2*z3*z4*z5
    sl5phi4 = sl5psi4.subs(h1, h1 + 1).subs(h2, 1).subs(h3, 1).subs(h4, h4 + 1).expand()

    g1 = groebner([sl5phi1], domain="QQ")
    g2 = groebner([sl5phi1, sl5phi2], domain="QQ")
    g3 = groebner([sl5phi1, sl5phi2, sl5phi3], domain="QQ")

    print("sl5phi1:", sl5phi1)
    print("sl5phi2:", sl5phi2)
    print("NEW: sl5phi3", sl5phi3)

    # And now in "full" generality

    i, j, k, n = symbols("i j k n")
    h = IndexedBase("h")
    #z = IndexedBase("z")
    z1 = n * h[1] + h[n] + (n + 1) * (n - 2) / 2
    a = -h[1] + h[n] + (n + 1) * (n - 2) / 2
    # sumzi = (n - 1) * (-h[1] + h[n])  # i = 2 to n
    sumzi = Sum(a - (i-2)*(n+1), (i, 2, n)).doit()
    znplus1 = -z1 - sumzi
    sumzizj = Sum((a - (i - 2) * (n + 1)) * (a - (j - 2) * (n + 1)), (j, i + 1, n), (i, 2, n - 1)).doit().expand()
    sumzizjzk = Sum((a - (i - 2) * (n + 1)) * (a - (j - 2) * (n + 1)) * (a - (k - 2) * (n + 1)), (k, j + 1, n),
                    (j, i + 1, n - 1), (i, 2, n - 2)).doit().expand()
    # # othersumzizj = (a ** 2 * (n - 1) * (n - 2) / 2 - a * (n + 1) * (n - 2) ** 2 * (n - 1) / 2 + (n + 1) ** 2 * (n - 3) * (n - 2) * (
    # #             n - 1) * (3 * n - 4) / 24) # This should equal sumzizj
    psi1 = z1 * znplus1 + (z1 + znplus1) * sumzi + sumzizj
    phi1 = psi1.subs(h[1], h[1]+1).subs(h[n], h[n]+1).as_poly(h[1], h[n])

    psi2 = z1*znplus1 * sumzi + (z1 + znplus1) * sumzizj + sumzizjzk
    phi2 = psi2.subs(h[1], h[1]+1).subs(h[n], h[n]+1).as_poly(h[1], h[n])
    # # print(factor(Sum(i - 2 + j - 2, (j, i + 1, n), (i, 2, n - 1)).doit().expand()))
    # # print(factor(Sum((i - 2) * (j - 2), (j, i + 1, n), (i, 2, n - 1)).doit().expand()))

    f = Function("f")
    w1 = Wild("w1")
    w2 = Wild("w2")
    replacer = lambda e: e.func == f and (e.args[0] >= e.args[1] or e.args[1] >= e.args[2])
    ijk = Sum(f(i, j, k), (k, 4, n), (j, 3, n), (i, 2, n)).subs(n, 5).doit().replace(replacer, lambda e: 0)
    ijk.replace(f, lambda i, j, k: (a - (i - 2) * (n + 1)) * (a - (j - 2) * (n + 1)) * (a - (k - 2) * (n + 1)))

