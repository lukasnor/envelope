import functools
import itertools
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sympy import Rational, Matrix

from src.BasisVector import BasisVector
from src.Complex import Complex
from src.Element import Element, Replacement
from src.Monomial import Monomial
from src.Sum import Sum


# This function defines for a list of basis vectors their ad functions if their matrices describe the ad representation
def define_ad_action_functions_by_matices(vectors: List[BasisVector]):
    assert all(v.is_matrix for v in vectors)
    assert all(isinstance(v.matrix, np.ndarray) for v in vectors)
    for v in vectors:
        ad_v_dict: Dict[BasisVector, Element] = {}
        for w in vectors:
            ad_vw: np.ndarray = v.matrix @ w.matrix - w.matrix @ v.matrix
            ad_vw_element: Element = Monomial(Complex(0))
            for x in vectors:
                row, column = x.ones_index
                ad_vw_element += Monomial(Complex(ad_vw[row, column]), [(x, 1)])
                ad_vw -= ad_vw[row, column] * x.matrix
            ad_v_dict[w] = ad_vw_element.reduce()
        # This does not work, since ad_v has a bound variable ad_v_dict, which is inside of the for loop
        # After the for loop finishes, the value of ad_v_dict is the relevant value, if the function ad_v is called
        # def ad_v(w: BasisVector) -> Element:
        #     return ad_v_dict[w]
        # v.ad = ad_v

        # Proper closure with a value bound ad_v_dict
        v.ad = (lambda d: lambda w: d[w])(ad_v_dict)


def generate_sl(n: int) -> List[BasisVector]:
    basis = []
    index = 0
    # Ys
    for off_diagonal in range(1, n):
        for k in range(n - off_diagonal):
            row, column = k + off_diagonal, k
            e_ij = np.zeros((n, n))
            e_ij[row, column] = 1.
            # basis.append({"matrix": e_ij,
            #               "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
            #               "ones_index": (row, column)})
            basis.append(BasisVector(symbol="E_{" + str(row + 1) + str(column + 1) + "}",
                                     index=index,
                                     is_matrix=True,
                                     matrix=e_ij,
                                     ones_index=(row, column)
                                     )
                         )
            index += 1
    # Hs
    for i in range(0, n - 1):
        h_i = np.zeros((n, n))
        h_i[i, i], h_i[i + 1, i + 1] = 1., -1.
        # basis.append({"matrix": h_i,
        #               "symbol": "H_{" + str(i + 1) + "}",
        #               "ones_index": (i, i)})
        basis.append(BasisVector(symbol="H_{" + str(i + 1) + "}",
                                 index=index,
                                 is_matrix=True,
                                 matrix=h_i,
                                 ones_index=(i, i)
                                 )
                     )
        index += 1
    # Xs
    for off_diagonal in range(1, n):
        for k in range(n - off_diagonal):
            row, column = k, k + off_diagonal
            e_ij = np.zeros((n, n))
            e_ij[row, column] = 1.
            # basis.append({"matrix": e_ij,
            #               "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
            #               "ones_index": (row, column)})
            basis.append(BasisVector(symbol="E_{" + str(row + 1) + str(column + 1) + "}",
                                     index=index,
                                     is_matrix=True,
                                     matrix=e_ij,
                                     ones_index=(row, column)
                                     )
                         )
            index += 1
    define_ad_action_functions_by_matices(basis)
    # Give every vector its dual element wrt the killing form
    ads = generate_ad_action_matrices(basis)
    killing_form = np.array([[np.trace(ads[i] @ ads[j]) for j in range(len(basis))] for i in range(len(basis))],
                            dtype="int")
    killing_form = Matrix(killing_form)  # Convert to Sympy matrix, to get inverse with Rational coefficients
    k_inv = killing_form.inv()
    for i in range(len(basis)):
        basis[i].dual = Sum(*(Monomial(Complex(k_inv[j, i]), [(basis[j], 1)]) for j in range(len(basis)))).reduce()
    return basis


def generate_standard_sl(n: int):
    basis = []
    # Hs
    for i in range(0, n - 1):
        H_i = np.zeros((n, n))
        H_i[i, i], H_i[i + 1, i + 1] = 1., -1.
        basis.append({"matrix": H_i,
                      "symbol": "H_{" + str(i + 1) + "}",
                      "ones_index": (i, i)})
    # Xs
    for off_diagonal in range(1, n):
        for k in range(n - off_diagonal):
            row, column = k, k + off_diagonal
            E_ij = np.zeros(shape=(n, n))
            E_ij[row, column] = 1.
            basis.append({"matrix": E_ij,
                          "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
                          "ones_index": (row, column)})
    # Ys
    for off_diagonal in range(1, n):
        for k in range(n - off_diagonal):
            row, column = k + off_diagonal, k
            E_ij = np.zeros(shape=(n, n))
            E_ij[row, column] = 1.
            basis.append({"matrix": E_ij,
                          "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
                          "ones_index": (row, column)})
    return basis


def generate_ad_action_matrices(vectors: List[BasisVector]) -> List[np.ndarray]:
    assert all(v.is_matrix for v in vectors)
    ad_matrices = []
    for i in range(len(vectors)):
        ad_i = np.zeros(shape=(len(vectors), len(vectors)))
        for column in range(len(vectors)):
            m = vectors[i].matrix @ vectors[column].matrix \
                - vectors[column].matrix @ vectors[i].matrix
            for row in range(len(vectors)):
                ad_i[row, column] = m[vectors[row].ones_index]
                m = m - ad_i[row, column] * vectors[row].matrix  # maybe this fixes it
        ad_matrices.append(ad_i)
    return ad_matrices


def ad_dict(vectors: List[BasisVector]) -> Dict[BasisVector, Dict[BasisVector, Element]]:
    ad_dict = {}
    for v in vectors:
        v_dict = {}
        for w in vectors:
            v_dict[w] = v.ad(w)
        ad_dict[v] = v_dict
    return ad_dict


# For a basis of the form [a, b, c, ..., da, db, dc, ...]
def define_ad_action_for_differential_operator_basis(basis):
    assert len(basis) % 2 == 0
    m = int(len(basis) / 2)
    for v in basis:
        ad_v_dict = {}
        for w in basis:
            if w.index == m + v.index:
                ad_v_dict[w] = Monomial(Complex(-1))
            elif w.index == v.index - m:
                ad_v_dict[w] = Monomial(Complex(1))
            else:
                ad_v_dict[w] = Monomial(Complex(0))
        v.ad = (lambda d: lambda w: d[w])(ad_v_dict)


def generate_z_basis(n: int):
    # 4*n elements in basis
    basis: List[BasisVector] = []
    for i in range(n):
        zi = BasisVector(symbol="z_{" + str(i + 1) + "}",
                         index=i,
                         is_matrix=False)
        zibar = BasisVector(symbol="\\bar{z}_{" + str(i + 1) + "}",
                            index=n + i,
                            is_matrix=False)
        dzi = BasisVector(symbol="\\frac{\\partial}{\\partial z_{" + str(i + 1) + "}}",
                          index=2 * n + i,
                          is_matrix=False)
        dzibar = BasisVector(symbol="\\frac{\\partial}{\\partial \\bar{z}_{" + str(i + 1) + "}}",
                             index=3 * n + i,
                             is_matrix=False)
        basis += [zi, zibar, dzi, dzibar]
    basis.sort()
    define_ad_action_for_differential_operator_basis(basis)
    return basis


def generate_xy_basis(n: int):
    basis: List[BasisVector] = []
    for i in range(n):
        xi = BasisVector(symbol="x_{" + str(i + 1) + "}",
                         index=i,
                         is_matrix=False)
        yi = BasisVector(symbol="y_{" + str(i + 1) + "}",
                         index=i + n,
                         is_matrix=False)
        dxi = BasisVector(symbol="\\frac{\\partial}{\\partial x_{" + str(i + 1) + "}}",
                          index=i + 2 * n,
                          is_matrix=False)
        dyi = BasisVector(symbol="\\frac{\\partial}{\\partial y_{" + str(i + 1) + "}}",
                          index=i + 3 * n,
                          is_matrix=False)
        basis += [xi, yi, dxi, dyi]
    basis.sort()
    define_ad_action_for_differential_operator_basis(basis)
    return basis


def generate_z_by_xy_replacement(z_basis: List[BasisVector], xy_basis: List[BasisVector]) -> Replacement:
    assert len(z_basis) == len(xy_basis)
    # assert z_basis.is_sorted() and xy_basis.is_sorted()
    assert len(z_basis) % 4 == 0
    n = int(len(z_basis) / 4)
    replacement = {}
    I = Complex(0, 1)
    half = Complex(Rational(1, 2))
    for j in range(n):
        xj = Monomial.convert(xy_basis[j])
        yj = Monomial.convert(xy_basis[n + j])
        dxj = Monomial.convert(xy_basis[2 * n + j])
        dyj = Monomial.convert(xy_basis[3 * n + j])
        # zj -> xj + i yj
        replacement[z_basis[j]] = xj + I * yj
        # zjbar -> xj - i yj
        replacement[z_basis[j + n]] = xj - I * yj
        # dzj -> 1/2 (dxj - i dyj)
        replacement[z_basis[j + 2 * n]] = half * (dxj - I * dyj)
        # dzjbar -> 1/2 (dxj + i dyj)
        replacement[z_basis[j + 3 * n]] = half * (dxj + I * dyj)
    return replacement


def matrix_stuff():
    ads = generate_ad_action_matrices(classical_basis)
    k = np.array(
        [[np.trace(ads[i] @ ads[j]) for j in range(len(classical_basis))] for i in range(len(classical_basis))],
        dtype="int")
    K = Matrix(k)
    a = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 2., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 3., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 2., 0., 0., 0., 0., 0.]])
    b = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 2., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 3., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 2., 0.]])
    c = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                  [3., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
                  [0., 2., 0., 0., 0., 0., 0., 0., 0., 0.]])


def casimir_second_order() -> Element:
    # 'Dual' elements
    H1 = Complex(Rational(1, 9)) * h1 + Complex(Rational(1, 18)) * h2
    H2 = Complex(Rational(1, 18)) * h1 + Complex(Rational(1, 9)) * h2
    sixth = Complex(Rational(1, 6))
    E12 = sixth * e21
    E23 = sixth * e32
    E13 = sixth * e31
    E21 = sixth * e12
    E32 = sixth * e23
    E31 = sixth * e13

    casimir_2 = 12 * H1 * H1 - 6 * H1 * H2 - 6 * H2 * H1 + 12 * H2 * H2 + \
                6 * (E12 * E21 + E21 * E12 + E13 * E31 + E31 * E13 + E23 * E32 + E32 * E23)
    return casimir_2.reduce().canonicalize().sort_by_degree()


def casimir_third_order() -> Element:
    # 'Dual' elements
    H1 = Complex(Rational(1, 9)) * h1 + Complex(Rational(1, 18)) * h2
    H2 = Complex(Rational(1, 18)) * h1 + Complex(Rational(1, 9)) * h2
    sixth = Complex(Rational(1, 6))
    E12 = sixth * e21
    E23 = sixth * e32
    E13 = sixth * e31
    E21 = sixth * e12
    E32 = sixth * e23
    E31 = sixth * e13

    casimir_3 = 10 * (E12 * E23 * E31 - E12 * H1 * E21 + E12 * H2 * E21 + E12 * E21 * H1
                      - E23 * E32 * H1 + E23 * E32 * H2 + E23 * E31 * E12 - E23 * H2 * E32
                      + E13 * E32 * E21 + E13 * E31 * H1 - E13 * H2 * E31
                      - E21 * E12 * H1 + E21 * E12 * H2 + E21 * E13 * E32 + E21 * H1 * E12
                      - E32 * E23 * H2 + E32 * E21 * E13 - E32 * H1 * E23 + E32 * H2 * E23
                      + E31 * E12 * E23 - E31 * E13 * H2 + E31 * H1 * E13
                      + H1 * E12 * E21 - H1 * E23 * E32 + H1 * E13 * E31 - H1 * E21 * E12 + H1 * H1 * H2 + H1 * H2 * H1 - H1 * H2 * H2
                      + H2 * E23 * E32 + H2 * E21 * E12 - H2 * E32 * E23 - H2 * E31 * E13 - H2 * H1 * H2 + H2 * H1 * H1 - H2 * H2 * H1)
    return casimir_3.reduce().canonicalize().sort_by_degree()


def casimir(order: int, basis: List[BasisVector]) -> Element:
    xyzs = itertools.product(basis, repeat=order)
    matrix = lambda x: x.matrix  # TODO: Give the option to use an arbitrary representation, not just the standard one
    tr = lambda xyz: Complex(np.trace(functools.reduce(np.ndarray.__matmul__, map(matrix, xyz))))
    dual = lambda x: x.dual
    dual_product = lambda xyz: functools.reduce(Element.__mul__, map(dual, xyz))
    summand = lambda xyz: tr(xyz) * dual_product(xyz)
    normalisation = 2 * (basis[0].matrix.shape[0])  # 2 * dimension of the representation
    casimir = functools.reduce(Element.__add__, map(summand, xyzs))
    return casimir.reduce().canonicalize().normalize().sort_by_degree()


# TODO: This hangs by a thread
def cut_off_casimir(cc, z_replacement, xy_replacement):
    from src.Sum import Sum
    p = Sum(*cc.summands[-11:]).sort_by_degree()
    print(p)
    t = Sum(*p.summands[:4])
    print(t)
    tz = t.replace(z_replacement).reduce().canonicalize().sort_by_degree()
    print(tz)
    tz_prime = Sum(*tz.summands[:56])
    print(tz_prime)
    txy = tz_prime.replace(xy_replacement).reduce().canonicalize().sort_by_degree().group_by_coefficient()
    print(txy)


if __name__ == "__main__":
    n = 2
    classical_basis = generate_sl(n)
    c2 = casimir(order=2, basis=classical_basis)
    print("2nd order Casimir in sl_2")
    print(c2)
    print()

    n = 3
    classical_basis = generate_sl(n)
    # Regular elements
    # Ys, Hs, Xs
    e21, e32, e31, h1, h2, e12, e23, e13 = tuple(map(Monomial.convert, classical_basis))

    c2 = casimir(order=2, basis=classical_basis)
    print("2nd order Casimir in sl_3")
    print(c2)
    print()
    c3 = casimir(order=3, basis=classical_basis)
    print("3rd order Casimir in sl_3")
    print(c3)
    print()

    print("Test, ob beide Methoden zum Erzeugen der Casimirs das gleiche tun:")
    print("C2:", (c2 - 9*casimir_second_order()).reduce())
    print("C3:", (5*c3 - 2*486*casimir_third_order()).reduce())  # At least up to a factor
    print()

    # z and z bar elements
    z_basis = generate_z_basis(n)
    z1, z2, z3, \
        z1bar, z2bar, z3bar, \
        dz1, dz2, dz3, \
        dz1bar, dz2bar, dz3bar = tuple(map(Monomial.convert, z_basis))

    # How do the sl_3 elements act on complex homogeneous polynomials
    z_replacement: Replacement = {
        classical_basis[0]: - z1 * dz2 + z2bar * dz1bar,  # Y_1
        classical_basis[1]: - z2 * dz3 + z3bar * dz2bar,  # Y_2
        classical_basis[2]: - z1 * dz3 + z3bar * dz1bar,  # Y_3
        classical_basis[3]: - z1 * dz1 + z2 * dz2 + z1bar * dz1bar - z2bar * dz2bar,  # H_1
        classical_basis[4]: - z2 * dz2 + z3 * dz3 + z2bar * dz2bar - z3bar * dz3bar,  # H_2
        classical_basis[5]: - z2 * dz1 + z1bar * dz2bar,  # X_1
        classical_basis[6]: - z3 * dz2 + z2bar * dz3bar,  # X_2
        classical_basis[7]: - z3 * dz1 + z1bar * dz3bar  # X_3
    }

    # x and y elements
    xy_basis = generate_xy_basis(n)
    xy_replacement = generate_z_by_xy_replacement(z_basis, xy_basis)

    # Casimirs in xy coordinates
    # c2z = c2.replace(z_replacement).reduce().canonicalize()
    # c2xy = c2z.replace(xy_replacement).reduce().canonicalize().sort_by_degree().group_by_coefficient()
    # c3z = c3.replace(z_replacement).reduce().canonicalize()
    # c3xy = c3z.replace(xy_replacement).reduce().canonicalize().sort_by_degree().group_by_coefficient()
    # c3xy_corrected = (c3xy + Rational(1, 4) * c2xy).reduce().group_by_coefficient()

    # l = []
    # for v in classical_basis:
    #     for w in classical_basis:
    #         for x in classical_basis:
    #             tr = np.trace(v.matrix @ w.matrix @ x.matrix)
    #             if not tr == 0:
    #                 #print(v, w, x, ":", tr)
    #                 l.append(tr)
    # print(len(l))
    #
    # ads = generate_ad_action_matrices(classical_basis)
    # c = Monomial(Complex(0))
    # for v in classical_basis:
    #     for w in classical_basis:
    #         for x in classical_basis:
    #             tr = np.trace(ads[v.index] @ ads[w.index] @ ads[x.index])
    #             if not tr == 0:
    #                 print(v, w, x, ":", Complex(tr))
    #                 c += Complex(tr) * v.dual * w.dual * x.dual
    # print(c.reduce())
    # c = (-4 * c).reduce().canonicalize().group_by_coefficient()

    one = Complex(1)
    One = Monomial(one)
    projection_replacement = defaultdict(lambda: Monomial(Complex(0)), {classical_basis[3]: h1, classical_basis[4]: h2})
    chandra_replacement = {classical_basis[3]: h1 - One, classical_basis[4]: h2 - One}
    alpha_1_replacement = {classical_basis[3]: -h1, classical_basis[4]: h1 + h2}
    alpha_2_replacement = {classical_basis[3]: h1 + h2, classical_basis[4]: -h2}

    d3 = c3.replace(projection_replacement).reduce()
    print("This element should only contain H factors:")
    print(d3)
    d3sym = (d3 + 9*c2).replace(projection_replacement).reduce().sort_by_degree()
    print("Or even more symmetric:")
    print(d3sym)
    print()
    e = d3sym.replace(chandra_replacement).reduce().canonicalize().sort_by_degree()
    print("This is the supersymmetric degree 3 polynomial in H1 and H2:")
    print(e)
    print()
    f = (e.replace(alpha_1_replacement) - e).reduce().canonicalize()
    print("Check, if it is invariant under root reflection. Expected results: 0.")
    print("alhpa_1:", f)
    f = (e.replace(alpha_2_replacement) - e).reduce().canonicalize()
    print("alpha_2:", f)
