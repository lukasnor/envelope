from sympy import *

if __name__ == "__main__":
    w, z, dw, dz = map(IndexedBase,
                       ['w', 'z', 'dw', 'dz'])
    i = Idx('i', (1, 3))
    j = Wild('j')
    x, y, dx, dy = map(IndexedBase, ['x', 'y', 'dx', 'dy'])
    H = IndexedBase('H')
    X, Y = map(IndexedBase, ['X', 'Y'])
    H1 = -1 * w[1] * dw[1] + w[2] * dw[2] + z[1] * dz[1] - z[2] * dz[2]
    H2 = -1 * w[2] * dw[2] + w[3] * dw[3] + z[2] * dz[2] - z[3] * dz[3]
    H3 = H1 + H2
    X1 = -w[2] * dw[1] + z[1] * dz[2]
    X2 = -w[3] * dw[2] + z[2] * dz[3]
    X3 = -w[3] * dw[1] + z[1] * dz[3]
    Y1 = -w[1] * dw[2] + z[2] * dz[1]
    Y2 = -w[2] * dw[3] + z[3] * dz[2]
    Y3 = -w[1] * dw[3] + z[3] * dz[1]

    noncommutativeAdjustmentsH = Sum(w[i] * dw[i] + z[i] * dz[i], (i, 1, 3)).doit()
    noncommutativeAdjustmentsXY = 2 * w[1] * dw[1] + w[2] * dw[2] + z[2] * dz[2] + 2 * z[3] * dz[3]

    casimir = Rational(1, 9) * (H[1] ** 2 + H[2] ** 2 + H[1] * H[2] + noncommutativeAdjustmentsH) \
              + Rational(1, 3) * (Y[1] * X[1] + Y[2] * X[2] + Y[3] * X[3] + noncommutativeAdjustmentsXY + H[3])

    casimir_wz = casimir.subs({H[1]: H1, H[2]: H2, H[3]: H3,
                               X[1]: X1, X[2]: X2, X[3]: X3,
                               Y[1]: Y1, Y[2]: Y2, Y[3]: Y3})

    casimir_xy = casimir_wz.replace(w[j], x[j] + I * y[j]). \
        replace(dw[j], Rational(1, 2) * (dx[j] - I * dy[j])). \
        replace(z[j], x[j] - I * y[j]). \
        replace(dz[j], Rational(1, 2) * (dx[j] + I * dy[j])).expand()

    print(latex(casimir_xy.as_poly(x[j], y[j], dx[j], dy[j])))
    casimir_xy.as_poly(x[1], x[2], x[3], y[1], y[2], y[3], dx[1], dx[2], dx[3], dy[1], dy[2], dy[3])