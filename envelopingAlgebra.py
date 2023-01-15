from typing import Dict, List

import numpy as np
from sympy import Rational, Matrix

from src.BasisVector import BasisVector
from src.Complex import Complex
from src.Element import Element
from src.Monomial import Monomial


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


def define_ad_action_for_zs(basis):
    assert len(basis) % 4 == 0
    n = int(len(basis) / 4)
    for v in basis:
        ad_v_dict = {}
        for w in basis:
            if w.index == 2 * n + v.index:
                ad_v_dict[w] = Monomial(Complex(-1))
            elif w.index == v.index - 2 * n:
                ad_v_dict[w] = Monomial(Complex(1))
            else:
                ad_v_dict[w] = Monomial(Complex(0))
        v.ad = (lambda d: lambda w: d[w])(ad_v_dict)


def generate_z_basis(n: int):
    # 4*n elements in basis
    basis: List[BasisVector] = []
    for i in range(n):
        z_i = BasisVector(symbol="z_{" + str(i + 1) + "}",
                          index=i,
                          is_matrix=False)
        z_i_bar = BasisVector(symbol="\\bar{z}_{" + str(i + 1) + "}",
                              index=n + i,
                              is_matrix=False)
        dz_i = BasisVector(symbol="\\frac{\\partial}{\\parial z_{" + str(i + 1) + "}}",
                           index=2 * n + i,
                           is_matrix=False)
        dz_i_bar = BasisVector(symbol="\\frac{\\partial}{\\parial \\bar{z}_{" + str(i + 1) + "}}",
                               index=3 * n + i,
                               is_matrix=False)
        basis += [z_i, z_i_bar, dz_i, dz_i_bar]
    basis.sort()
    define_ad_action_for_zs(basis)
    return basis


def generate_xy_basis(n: int):
    pass


def generate_ad_action(xy_basis):
    pass


def reduced_casimir_second_order():
    casimir_2 = 12 * H1 * H1 - 6 * H1 * H2 - 6 * H2 * H1 + 12 * H2 * H2 + \
                6 * (E12 * E21 + E21 * E12 + E13 * E31 + E31 * E13 + E23 * E32 + E32 * E23)
    reduced_casimir_2 = casimir_2.reduce()
    print(reduced_casimir_2)
    return reduced_casimir_2


def reduced_casimir_third_order():
    casimir_3 = 10 * (E12 * E23 * E31 - E12 * H1 * E21 + E12 * H2 * E21
                      - E23 * E32 * H1 + E23 * E32 * H2 + E23 * E31 * E12 - E23 * H2 * E32
                      + E12 * E32 * E21 + E13 * E31 * H1 - E13 * H2 * E31
                      - E21 * E12 * H1 + E21 * E12 * H2 + E21 * E13 * E32 + E21 * H1 * E12
                      - E32 * E23 * H2 + E32 * E21 * E13 - E32 * H1 * E23 + E32 * H2 * E23
                      + E31 * E12 * E23 - E31 * E13 * H2 + E31 * H1 * E13
                      + H1 * E12 * E21 - H1 * E23 * E32 + H1 * E13 * E31 - H1 * E21 * E12 + H1 * H1 * H2 + H1 * H2 * H1 - H1 * H2 * H2
                      + H2 * E23 * E32 + H2 * E21 * E12 - H2 * E32 * E23 - H2 * E31 * E13 - H2 * H1 * H2 + H2 * H1 * H1 - H2 * H2 * H1)
    reduced_casimir_3 = casimir_3.reduce()
    print(reduced_casimir_3)
    return reduced_casimir_3


if __name__ == "__main__":
    n = 3
    classical_basis = generate_sl(n)

    # Regular elements
    # Ys
    e21 = Monomial(Complex(1), [(classical_basis[0], 1)])
    e32 = Monomial(Complex(1), [(classical_basis[1], 1)])
    e31 = Monomial(Complex(1), [(classical_basis[2], 1)])
    # Hs
    h1 = Monomial(Complex(1), [(classical_basis[3], 1)])
    h2 = Monomial(Complex(1), [(classical_basis[4], 1)])
    # Xs
    e12 = Monomial(Complex(1), [(classical_basis[5], 1)])
    e23 = Monomial(Complex(1), [(classical_basis[6], 1)])
    e13 = Monomial(Complex(1), [(classical_basis[7], 1)])

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

    ads = generate_ad_action_matrices(classical_basis)
    k = np.array(
        [[np.trace(ads[i] @ ads[j]) for j in range(len(classical_basis))] for i in range(len(classical_basis))],
        dtype="int")
    K = Matrix(k)
    # e = e12 * e12 * h1
    # print(e)
    # print(e.canonicalize())
    c = reduced_casimir_third_order()
    cc = c.canonicalize()
    print(cc)
    # one = Complex(1, 0)
    # i = Complex(0, 1)
    # e = one + i*one
    # b = Complex(1, 1)
    # print(b == e)
    # print(e**2)
    # print(3 * e)
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

    z_basis = generate_z_basis(n)
    z1 = Monomial(Complex(1), [(z_basis[0], 1)])
    z2 = Monomial(Complex(1), [(z_basis[1], 1)])
    z3 = Monomial(Complex(1), [(z_basis[2], 1)])
    z1bar = Monomial(Complex(1), [(z_basis[3], 1)])
    z2bar = Monomial(Complex(1), [(z_basis[4], 1)])
    z3bar = Monomial(Complex(1), [(z_basis[5], 1)])
    dz1 = Monomial(Complex(1), [(z_basis[6], 1)])
    dz2 = Monomial(Complex(1), [(z_basis[7], 1)])
    dz3 = Monomial(Complex(1), [(z_basis[8], 1)])
    dz1bar = Monomial(Complex(1), [(z_basis[9], 1)])
    dz2bar = Monomial(Complex(1), [(z_basis[10], 1)])
    dz3bar = Monomial(Complex(1), [(z_basis[11], 1)])
    Replacement = Dict[BasisVector, Element]
    replacement: Replacement = {
        classical_basis[0]: - z1 * dz2 + z2bar * dz1bar,  # Y_1
        classical_basis[1]: - z2 * dz3 + z3bar * dz2bar,  # Y_2
        classical_basis[2]: - z1 * dz3 + z3bar * dz1bar,  # Y_3
        classical_basis[3]: - z1 * dz1 + z2 * dz2 + z1bar * dz1bar - z2bar * dz2bar,  # H_1
        classical_basis[4]: - z2 * dz2 + z3 * dz3 + z2bar * dz2bar - z3bar * dz3bar,  # H_2
        classical_basis[5]: - z2 * dz1 + z1bar * dz2bar,  # X_1
        classical_basis[6]: - z3 * dz2 + z2bar * dz3bar,  # X_2
        classical_basis[7]: - z3 * dz1 + z1bar * dz3bar  # X_3
    }
