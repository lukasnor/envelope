import abc
import copy
import functools
import numbers
from abc import abstractmethod
from functools import reduce
from collections import Counter, defaultdict
from typing import Iterable, Tuple, Hashable, overload, Dict, List
import numpy as np
from sympy import Rational, latex, sqrt


class Complex:

    def __init__(self, re, im=0.0):
        self.re = Rational(re)
        self.im = Rational(im)

    def __add__(self, other):
        return Complex(self.re + other.re, self.im + other.im)

    def __sub__(self, other):
        return self + (- other)

    def __neg__(self):
        return Complex(-1) * self

    def __mul__(self, other):
        if isinstance(other, Complex):
            return Complex(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)
        elif isinstance(other, Rational):
            return self * Complex(other, 0)
        elif isinstance(other, numbers.Complex) and \
                isinstance(other.real, numbers.Rational) and \
                isinstance(other.imag, numbers.Rational):
            return self * Complex(Rational(other.real.numerator, other.real.denominator),
                                  Rational(other.imag.numerator, other.imag.denominator))
        elif isinstance(other, numbers.Rational):
            return self * Complex(Rational(other.numerator, other.denominator))
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Rational):
            return self * Complex(other)
        elif isinstance(other, numbers.Rational):
            return self * Complex(Rational(other.numerator, other.denominator), 0)
        elif isinstance(other, numbers.Complex) and \
                isinstance(other.real, numbers.Rational) and \
                isinstance(other.imag, numbers.Rational):
            return self * Complex(Rational(other.real.numerator, other.real.denominator),
                                  Rational(other.imag.numerator, other.imag.denominator))
        else:
            return NotImplemented

    def __pow__(self, power: int):
        return reduce(Complex.__mul__, [self for _ in range(power)])

    def __eq__(self, other):
        if isinstance(other, Complex):
            return (self.re, self.im) == (other.re, other.im)
        elif isinstance(other, Rational):
            return self == Complex(other, 0)
        elif isinstance(other, numbers.Complex):
            return self == Complex(other.real, other.imag)
        else:
            return NotImplemented

    def __str__(self):
        if self.im == Rational(0):
            return latex(self.re)
        if self.re == Rational(0):
            if self.im == Rational(1):
                return "i"
            return latex(self.im) + " i"
        if self.im == Rational(1):
            return latex(self.re) + " + i"
        return latex(self.re) + " + " + latex(self.im) + " i"

    def __abs__(self):
        return Complex(sqrt(self.re ** 2 + self.im ** 2), 0)

    def real(self):
        return Complex(self.re, 0)

    def imag(self):
        return Complex(self.im, 0)

    def conjugate(self):
        return Complex(self.re, - self.im)


class Element(abc.ABC):

    def __init__(self):
        self.is_reduced = self._determine_reduced()

    def __add__(self, other: 'Element') -> 'Element':
        return Sum(self, other)

    def __neg__(self):
        return Complex(-1) * self

    def __sub__(self, other: 'Element') -> 'Element':
        return self + - other

    def __mul__(self, other: 'Element') -> 'Element':
        return Product(self, other)

    def __rmul__(self, other):
        if isinstance(other, Complex):
            return Monomial(other) * self
        elif isinstance(other, Rational):
            return Monomial(Complex(other)) * self
        elif isinstance(other, numbers.Complex) \
                and isinstance(other.real, numbers.Rational) \
                and isinstance(other.imag, numbers.Rational):
            return Monomial(Complex(other.real, other.imag)) * self
        else:
            return NotImplemented

    def __pow__(self, power, modulo=None):
        return Product(*(self for _ in range(power)))

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def signature(self) -> Hashable:
        ...

    @abstractmethod
    def _determine_reduced(self) -> bool:
        ...

    @abstractmethod
    def reduce(self) -> 'Element':
        ...

    @abstractmethod
    def canonicalize(self) -> 'Element':
        ...

    # @abstractmethod
    # def replace(self) -> 'Element': ...


class Monomial(Element):

    def __init__(self, coefficient: Complex, simple_factors: Iterable[Tuple['BasisVector', int]] = tuple([])):
        self.coefficient = coefficient
        self.simple_factors: Tuple[Tuple['BasisVector', int], ...] = tuple(simple_factors)
        super().__init__()

    @overload
    def __mul__(self, other: 'Monomial') -> 'Monomial':
        ...

    def __mul__(self, other):
        if isinstance(other, Monomial):
            return Monomial(self.coefficient * other.coefficient, self.simple_factors + other.simple_factors)
        else:
            return super().__mul__(other)

    # This is not the best way to print the monomials, since this depends on the variable "basisVectors"
    # from outer scope. But I don't want to introduce the basis to each element.
    def __str__(self):
        string = ""
        for vector, exponent in self.simple_factors:
            if exponent == 1:
                string += vector.symbol
            else:
                string += vector.symbol + "^{" + str(exponent) + "}"
        if self.coefficient == Complex(1) and string != "":
            return string
        if string == "":
            return str(self.coefficient)
        return str(self.coefficient) + " " + string

    # Equality is specific to Monomials
    # TODO: Is this necessary?
    def __eq__(self, other):
        #  print("This is apparently meant to mean mathematical equality")
        if not isinstance(other, Monomial):
            return False
        return self.coefficient == other.coefficient and self.reduce().signature() == other.reduce().signature()


    def signature(self) -> Tuple[Tuple['BasisVector', int]]:
        if not self.is_reduced:
            raise Exception("Using the signature of a non reduced element is not good.")
        return self.simple_factors

    def _determine_reduced(self):
        # Canonical form for any scalar c, including 0 is Monomial(coefficient=c, simple_factors=())
        if self.coefficient == 0 and len(self.simple_factors) > 0:
            return False
        if any(exponent == 0 for _, exponent in self.simple_factors):
            return False
        for i in range(len(self.simple_factors) - 1):
            assert isinstance(self.simple_factors[i][0], BasisVector)
            if self.simple_factors[i][0] == self.simple_factors[i + 1][0]:  # Are following simple factors the same
                return False
        return True

    @overload
    def reduce(self) -> 'Monomial':
        ...

    def reduce(self):
        if self.is_reduced:
            return self
        if self.coefficient == 0:
            return Monomial(Complex(0))
        new_factors = []
        current_vector: BasisVector = self.simple_factors[0][0]
        current_exponent: int = 0
        for vector, exponent in self.simple_factors:
            if vector == current_vector:
                current_exponent += exponent
            else:
                if current_exponent >= 1:
                    new_factors.append((current_vector, current_exponent))
                current_vector, current_exponent = vector, exponent
        if current_exponent >= 1:
            new_factors.append((current_vector, current_exponent))
        return Monomial(self.coefficient, tuple(new_factors))

    def canonicalize(self):
        if not self.is_reduced:
            return self.reduce().canonicalize()
        if len(self.simple_factors) < 2:
            return self
        factor_index, not_determined = 0, True
        for i in range(len(self.simple_factors) - 1):
            # Here the ordering of BasisVector via the index is relevant
            # The latter comparison should be between BasisVector
            if not_determined and self.simple_factors[i][0] > self.simple_factors[i + 1][0]:
                factor_index, not_determined = i, False
                break  # an index has been found
        if not_determined:
            return self
        new_factors: [Element] = [Monomial(self.coefficient, self.simple_factors[:factor_index])]
        # A_m^n B_i^k = B_i A_m^n B_i^k-1 - (\sum_l=0^n A_m^l ad[i][m] A_m_n-1-l) B_i^k-1
        m, n = self.simple_factors[factor_index]  # TODO: Change the variable names to better fit their types
        i, k = self.simple_factors[factor_index + 1]
        new_factor = Monomial(Complex(1), [(i, 1), (m, n), (i, k - 1)]).reduce()
        # sum = Sum(*(Monomial(Complex(1), [(m, l)]) * ad[i][m] * Monomial(Complex(1), [(m, n-1-l)]) for l in range(n))).reduce()
        sum = Sum(*(Monomial(Complex(1), [(m, l)]) * i.ad(m) * Monomial(Complex(1), [(m, n - 1 - l)]) for l in
                    range(n))).reduce()
        new_factor -= sum * Monomial(Complex(1), [(i, k - 1)])
        new_factor = new_factor.reduce()
        # new_factor = Sum(*(Monomial(Rational(math.comb(k, j)), [(i, j)]) *  # (k over j) * H_i^j
        #                    (Monomial(Rational(-n)) * ad[i][m])**(k-j) *  # (-n * ad(H_i)(X_m))^(k-j)
        #                    Monomial(Rational(1), [(m, n)])  # X_m^n
        #                    for j in range(k+1))).reduce()
        new_factors.append(new_factor)
        new_factors.append(Monomial(Complex(1), self.simple_factors[factor_index + 2:]))
        element = Product(*new_factors).reduce()
        return element.canonicalize().reduce()

    def degree(self):
        return sum(exponent for _, exponent in self.simple_factors)


class Sum(Element):

    def __init__(self, *summands: Element):
        self.summands: list[Element] = list(summands)
        super().__init__()

    @overload
    def __add__(self, other: 'Sum') -> 'Sum':
        ...

    def __add__(self, other: Element):
        if isinstance(other, Sum):
            return Sum(*self.summands, *other.summands)
        else:
            return Sum(*self.summands, other)

    def __str__(self):
        return reduce(lambda a, b: a + " + " + b, map(str, self.summands))

    def signature(self):
        return tuple(summand.signature() for summand in self.summands)

    def _determine_reduced(self):
        # Check if empty or single element
        if len(self.summands) < 2:
            return False
        # Check that every summand is not a sum
        if any(isinstance(summand, Sum) for summand in self.summands):
            return False
        # Check that the unique summands are reduced
        if not all(summand.is_reduced for summand in self.summands):
            return False
        # Check that no zero is involved
        if any(summand == Monomial(Complex(0)) for summand in self.summands):
            return False
        # Check that no two summands of same signature exist, but are grouped with coefficient
        signatures = [summand.signature() for summand in self.summands]
        c = Counter(signatures)
        values = c.values()
        if any(value > 1 for value in values):
            return False
        return True

    def reduce(self):
        if self.is_reduced:
            return self
        if len(self.summands) == 0:
            return Monomial(Complex(0))
        # Check if single element
        if len(self.summands) == 1:
            return self.summands[0].reduce()
        # Check if summands are sums themselves and unpack
        if any(isinstance(summand, Sum) for summand in self.summands):
            new_summands = []
            for summand in self.summands:
                if isinstance(summand, Sum):
                    new_summands += summand.summands
                else:
                    new_summands.append(summand)
            return Sum(*new_summands).reduce()
        # Check that the summands are reduced
        if not all(summand.is_reduced for summand in self.summands):
            return Sum(*(summand.reduce() for summand in self.summands)).reduce()
        # Throw out all zeros
        if any(summand == Monomial(Complex(0)) for summand in self.summands):
            return Sum(*(summand for summand in self.summands if not summand == Monomial(Complex(0)))).reduce()
        # At this point, every summand should be a non-zero monomial
        if not all(isinstance(summand, Monomial) for summand in self.summands):
            raise Exception("In this almost reduced sum, every summand should be a monomial.")
        # Group monomials with the same signature
        # Use the signature as the key in a dict, and the coefficient as the value, adding coefficients of terms with
        # the same signature
        d: Dict[Tuple[Tuple['BasisVector', int]], Complex] = defaultdict(lambda: Complex(0))
        for summand in self.summands:
            assert isinstance(summand, Monomial)
            d[summand.signature()] += summand.coefficient
        return Sum(*(Monomial(coefficient, simple_factors) for simple_factors, coefficient in d.items())).reduce()

    def canonicalize(self):
        if not self.is_reduced:
            return self.reduce().canonicalize()
        return Sum(*map(lambda e: e.canonicalize(), self.summands)).reduce()


class Product(Element):

    def __init__(self, *factors: Element):
        self.factors: list[Element] = list(factors)
        super().__init__()

    @overload
    def __mul__(self, other: 'Product') -> 'Product':
        ...

    def __mul__(self, other):
        if isinstance(other, Product):
            return Product(*self.factors, *other.factors)
        return Product(*(self.factors + [other]))

    def __str__(self):
        return reduce(lambda a, b: "(" + a + ") (" + b + ")", map(str, self.factors))

    def signature(self):
        return tuple(factor.signature() for factor in self.factors)

    def _determine_reduced(self):
        return False

    def reduce(self):
        if self.is_reduced:
            return self
        # Check if empty product
        if len(self.factors) == 0:
            return Monomial(Rational(1), [])
        # Check if only one factor
        if len(self.factors) == 1:
            return self.factors[1].reduce()
        # Check if factors are products themselves and unpack
        if any(isinstance(factor, Product) for factor in self.factors):
            new_factors = []
            for factor in self.factors:
                if isinstance(factor, Product):
                    new_factors += factor.factors
                else:
                    new_factors.append(factor)
            #  [(f for f in factor.factors) if isinstance(factor, Product) else factor for factor in self.factors]
            return Product(*new_factors).reduce()
        # Check that the factors are reduced
        if not all(factor.is_reduced for factor in self.factors):
            return Product(*(factor.reduce() for factor in self.factors)).reduce()
        # At this point, every factor should be a monomial or a sum
        if not all(isinstance(factor, Monomial) or isinstance(factor, Sum) for factor in self.factors):
            raise Exception("In this not yet distributed product, every factor should be a monomial or a sum.")
        # Now distribute
        # There must be a better way to do this
        # TODO: Find the way
        list_of_summand_lists: [[Monomial]] = [[Monomial(Complex(1))]]
        for factor in self.factors:
            if isinstance(factor, Monomial):
                for summand_list in list_of_summand_lists:
                    summand_list.append(factor)
            else:
                assert isinstance(factor, Sum)
                new_list_of_summand_lists = []
                for summand_in_factor in factor.summands:
                    assert isinstance(summand_in_factor, Monomial)
                    list_of_summand_lists_copy = copy.deepcopy(list_of_summand_lists)
                    for summand_list in list_of_summand_lists_copy:
                        summand_list.append(summand_in_factor)
                    new_list_of_summand_lists += list_of_summand_lists_copy
                list_of_summand_lists = new_list_of_summand_lists
        real_summands = [*map(lambda summand: reduce(lambda a, b: a * b, summand), list_of_summand_lists)]
        return Sum(*real_summands).reduce()

    def canonicalize(self):
        if not self.is_reduced:
            return self.reduce().canonicalize().reduce()
        raise Exception("A product should never be reduced!")
        #  return Product(*map(lambda e: e.canonicalize(), self.factors)).reduce()


@functools.total_ordering  # Lazy, but computationally costly, implementation of le, ge, lt, gt, ne, eq
class BasisVector:

    # Each basis vector is determined by its index. Equality '==' and hashing depends on the index alone
    # The ad action has to be defined "manually" in post, once a complete basis is constructed,
    # for a canonicalization to work
    def __init__(self, symbol: str,
                 index: int,
                 is_matrix: bool = False,
                 matrix: np.ndarray = None,
                 ones_index: Tuple[int, int] = None):
        self.symbol = symbol
        self.index = index
        self.is_matrix = is_matrix
        self.matrix = matrix
        self.ones_index = ones_index
        if self.is_matrix:
            assert self.matrix is not None and self.ones_index is not None

    def __str__(self):
        return self.symbol

    def __eq__(self, other):
        if isinstance(other, BasisVector):
            return self.index == other.index
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, BasisVector):
            return self.index <= other.index
        else:
            return NotImplemented

    def __hash__(self):
        return self.index

    # This defines the ad action of each basis vector and is to be determined
    def ad(self, other: 'BasisVector') -> Element:
        pass


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
    # TODO: define ad at this point!
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
        # v.ad_dict = ad_v_dict


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
    define_ad_action_functions_by_matices(basis)

    # Regular elements
    # Ys
    e21 = Monomial(Complex(1), [(basis[0], 1)])
    e32 = Monomial(Complex(1), [(basis[1], 1)])
    e31 = Monomial(Complex(1), [(basis[2], 1)])
    # Hs
    h1 = Monomial(Complex(1), [(basis[3], 1)])
    h2 = Monomial(Complex(1), [(basis[4], 1)])
    # Xs
    e12 = Monomial(Complex(1), [(basis[5], 1)])
    e23 = Monomial(Complex(1), [(basis[6], 1)])
    e13 = Monomial(Complex(1), [(basis[7], 1)])

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

    # ads = generate_ad_action_matrices(basis)
    # k = np.array([[np.trace(ads[i] @ ads[j]) for j in range(len(basis))] for i in range(len(basis))])
    e = e12 * e12 * h1
    print(e)
    print(e.canonicalize())
    c = reduced_casimir_second_order()
    cc = c.canonicalize()
    print(cc)
    one = Complex(1, 0)
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
