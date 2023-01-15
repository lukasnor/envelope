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


def generate_z_basis(n: int):
    pass


def generate_ad_action_for_zs(z_basis):
    pass


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
    basis = generate_sl(n)

    # Regular elements
    # Ys
    e21 = basis[0]
    e32 = basis[1]
    e31 = basis[2]
    # Hs
    h1 = basis[3]
    h2 = basis[4]
    # Xs
    e12 = basis[5]
    e23 = basis[6]
    e13 = basis[7]

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

    ads = generate_ad_action_matrices(basis)
    k = np.array([[np.trace(ads[i] @ ads[j]) for j in range(len(basis))] for i in range(len(basis))], dtype="int")
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
    # replacement = Dict[int, Element]
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
