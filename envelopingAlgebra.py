import copy
import numbers
from abc import abstractmethod
from functools import reduce
from collections import Counter, defaultdict
from typing import Iterable, Tuple, Hashable, overload
import numpy as np
from sympy import Rational, latex, sqrt


def generate_sl(n: int):
    basis = []
    # Ys
    for off_dagiagonal in range(1, n):
        for k in range(n - off_dagiagonal):
            row, column = k + off_dagiagonal, k
            E_ij = np.zeros((n, n))
            E_ij[row, column] = 1.
            basis.append({"matrix": E_ij,
                          "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
                          "ones_index": (row, column)})
    # Hs
    for i in range(0, n - 1):
        H_i = np.zeros((n, n))
        H_i[i, i], H_i[i + 1, i + 1] = 1., -1.
        basis.append({"matrix": H_i,
                      "symbol": "H_{" + str(i + 1) + "}",
                      "ones_index": (i, i)})
    # Xs
    for off_dagiagonal in range(1, n):
        for k in range(n - off_dagiagonal):
            row, column = k, k + off_dagiagonal
            E_ij = np.zeros((n, n))
            E_ij[row, column] = 1.
            basis.append({"matrix": E_ij,
                          "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
                          "ones_index": (row, column)})
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
    for off_dagiagonal in range(1, n):
        for k in range(n - off_dagiagonal):
            row, column = k, k + off_dagiagonal
            E_ij = np.zeros(shape=(n, n))
            E_ij[row, column] = 1.
            basis.append({"matrix": E_ij,
                          "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
                          "ones_index": (row, column)})
    # Ys
    for off_dagiagonal in range(1, n):
        for k in range(n - off_dagiagonal):
            row, column = k + off_dagiagonal, k
            E_ij = np.zeros(shape=(n, n))
            E_ij[row, column] = 1.
            basis.append({"matrix": E_ij,
                          "symbol": "E_{" + str(row + 1) + str(column + 1) + "}",
                          "ones_index": (row, column)})
    return basis


def generate_ad_action_matrices(basisVectors):
    ad_matrices = []
    for i in range(len(basisVectors)):
        ad_i = np.zeros(shape=(len(basisVectors), len(basisVectors)))
        for column in range(len(basisVectors)):
            m = basisVectors[i]["matrix"] @ basisVectors[column]["matrix"] \
                - basisVectors[column]["matrix"] @ basisVectors[i]["matrix"]
            for row in range(len(basisVectors)):
                ad_i[row, column] = m[basisVectors[row]["ones_index"]]
                m = m - ad_i[row, column] * basisVectors[row]["matrix"]  # maybe this fixes it
        ad_matrices.append(ad_i)
    return ad_matrices


def generate_ad_action_matrix_index_to_element(basisVectors):
    ad = [[Monomial(Complex(0)) for _ in range(len(basisVectors))] for _ in range(len(basisVectors))]
    for i in range(len(basisVectors)):
        for j in range(len(basisVectors)):
            ad_ij = basisVectors[i]["matrix"] @ basisVectors[j]["matrix"] \
                    - basisVectors[j]["matrix"] @ basisVectors[i]["matrix"]
            for vector_index in range(len(basisVectors)):
                row, column = basisVectors[vector_index]["ones_index"]
                ad[i][j] += Monomial(Complex(ad_ij[row, column]), [(vector_index, 1)])
                ad_ij = ad_ij - ad_ij[row, column] * basisVectors[vector_index]["matrix"]
            ad[i][j] = ad[i][j].reduce()
    return ad


class Complex:

    def __init__(self, re, im=0.0):
        self.re = Rational(re)
        self.im = Rational(im)

    def __add__(self, other):
        return Complex(self.re + other.re, self.im + other.im)

    def __sub__(self, other):
        return Complex(self.re - other.re, self.im - other.im)

    def __mul__(self, other):
        if isinstance(other, Complex):
            return Complex(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)
        if isinstance(other, Rational):
            return self * Complex(other, 0)
        if isinstance(other, numbers.Complex) and \
                isinstance(other.real, numbers.Rational) and \
                isinstance(other.imag, numbers.Rational):
            return self * Complex(Rational(other.real.numerator, other.real.denominator),
                                  Rational(other.imag.numerator, other.imag.denominator))
        if isinstance(other, numbers.Rational):
            return self * Complex(Rational(other.numerator, other.denominator))
        raise NotImplementedError


    def __rmul__(self, other):
        if isinstance(other, Rational):
            return self * Complex(other)
        if isinstance(other, numbers.Rational):
            return self * Complex(Rational(other.numerator, other.denominator), 0)
        if isinstance(other, numbers.Complex) and \
                isinstance(other.real, numbers.Rational) and \
                isinstance(other.imag, numbers.Rational):
            return self * Complex(Rational(other.real.numerator, other.real.denominator),
                                  Rational(other.imag.numerator, other.imag.denominator))
        raise NotImplementedError

    def __pow__(self, power: int):
        return reduce(Complex.__mul__, [self for _ in range(power)])

    def __eq__(self, other):
        if isinstance(other, Complex):
            return (self.re, self.im) == (other.re, other.im)
        if isinstance(other, numbers.Complex):
            return self == Complex(other.real, other.imag)
        if isinstance(other, Rational):
            return self == Complex(other, 0)
        else:
            NotImplemented

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
        return Complex(sqrt(self.re**2 + self.im**2), 0)

    def real(self):
        return Complex(self.re, 0)

    def imag(self):
        return Complex(self.im, 0)

    def conjugate(self):
        return Complex(self.re, - self.im)


class Element:

    def __init__(self):
        self.is_reduced = self._determine_reduced()

    @abstractmethod
    def __add__(self, other: 'Element') -> 'Element':
        return Sum(self, other)

    @abstractmethod
    def __mul__(self, other: 'Element') -> 'Element':
        return Product(self, other)

    def __sub__(self, other: 'Element') -> 'Element':
        return self + Monomial(Rational(-1), []) * other

    def __pow__(self, power, modulo=None):
        return Product(*(self for _ in range(power)))

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __cmp__(self, other):
        pass

    @abstractmethod
    def signature(self) -> Hashable:
        pass

    @abstractmethod
    def _determine_reduced(self) -> bool:
        pass

    @abstractmethod
    def reduce(self) -> 'Element':
        pass

    @abstractmethod
    def canonicalize(self) -> 'Element':
        pass


class Monomial(Element):

    def __init__(self, coefficient: Complex, simple_factors: Iterable[Tuple[int, int]] = tuple([])):
        self.coefficient = coefficient
        self.simple_factors: Tuple[Tuple[int, int], ...] = tuple(simple_factors)
        super().__init__()

    def __eq__(self, other):
        if not isinstance(other, Monomial):
            return False
        return self.coefficient == other.coefficient and self.reduce().signature() == other.reduce().signature()

    def degree(self):
        return sum(exponent for _, exponent in self.simple_factors)

    def signature(self):
        if not self.is_reduced:
            raise Exception("Using the signature of a non reduced element is not good.")
        return self.simple_factors

    def __add__(self, other):
        return Sum(self, other)

    @overload
    def __mul__(self, other: 'Monomial') -> 'Monomial': ...

    def __mul__(self, other):
        if isinstance(other, Monomial):
            return Monomial(self.coefficient * other.coefficient, self.simple_factors + other.simple_factors)
        return Product(self, other)

    def __cmp__(self, other):
        pass

    def _determine_reduced(self):
        # Canonical form for any scalar a, including 0 is Monomial(coefficient=a, simple_factors=())
        if self.coefficient == 0 and len(self.simple_factors) > 0:
            return False
        if any(exponent == 0 for _, exponent in self.simple_factors):
            return False
        for i in range(len(self.simple_factors) - 1):
            if self.simple_factors[i][0] == self.simple_factors[i + 1][0]:
                return False
        return True

    def reduce(self) -> 'Monomial':
        if self.is_reduced:
            return self
        if self.coefficient == 0:
            return Monomial(Complex(0), [])
        new_factors = []
        current_index, current_exponent = self.simple_factors[0][0], 0
        for i in range(len(self.simple_factors)):
            index, exponent = self.simple_factors[i]
            if index == current_index:
                current_exponent += exponent
            else:
                if current_exponent >= 1:
                    new_factors.append((current_index, current_exponent))
                current_index, current_exponent = index, exponent
        if current_exponent >= 1:
            new_factors.append((current_index, current_exponent))
        return Monomial(self.coefficient, tuple(new_factors))

    def canonicalize(self):
        if not self.is_reduced:
            return self.reduce().canonicalize()
        if len(self.simple_factors) < 2:
            return self
        factor_index, not_determined = 0, True
        for i in range(len(self.simple_factors)-1):
            if not_determined and self.simple_factors[i][0] > self.simple_factors[i+1][0]:
                factor_index, not_determined = i, False
        if not_determined:
            return self
        new_factors: [Element] = []
        new_factors.append(Monomial(self.coefficient, self.simple_factors[:factor_index]))  # all factors up to the swap
        # A_m^n B_i^k = B_i A_m^n B_i^k-1 - (\sum_l=0^n A_m^l ad[i][m] A_m_n-1-l) B_i^k-1
        m, n = self.simple_factors[factor_index]
        i, k = self.simple_factors[factor_index+1]
        new_factor = Monomial(Complex(1), [(i, 1), (m, n), (i, k-1)]).reduce()
        sum = Sum(*(Monomial(Complex(1), [(m, l)]) * ad[i][m] * Monomial(Complex(1), [(m, n-1-l)]) for l in range(n))).reduce()
        new_factor -= sum * Monomial(Complex(1), [(i, k-1)])
        new_factor = new_factor.reduce()
        # new_factor = Sum(*(Monomial(Rational(math.comb(k, j)), [(i, j)]) *  # (k over j) * H_i^j
        #                    (Monomial(Rational(-n)) * ad[i][m])**(k-j) *  # (-n * ad(H_i)(X_m))^(k-j)
        #                    Monomial(Rational(1), [(m, n)])  # X_m^n
        #                    for j in range(k+1))).reduce()
        new_factors.append(new_factor)
        new_factors.append(Monomial(Complex(1), self.simple_factors[factor_index+2:]))
        element = Product(*new_factors).reduce()
        return element.canonicalize().reduce()

    def __str__(self):
        string = ""
        for index, exponent in self.simple_factors:
            if exponent == 1:
                string += basisVectors[index]["symbol"]
            else:
                string += basisVectors[index]["symbol"] + "^{" + str(exponent) + "}"
        if self.coefficient == Rational(1) and string != "":
            return string
        if string == "":
            return latex(self.coefficient)
        return latex(self.coefficient) + " " + string


class Sum(Element):

    def __init__(self, *summands: Element):
        self.summands: list[Element] = list(summands)
        super().__init__()

    @overload
    def __add__(self, other: 'Sum') -> 'Sum': ...

    def __add__(self, other: Element):
        if isinstance(other, Sum):
            return Sum(*self.summands, *other.summands)
        return Sum(*self.summands, other)

    def __mul__(self, other):
        return Product(*[self, other])

    def __cmp__(self, other):
        raise NotImplementedError

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
            return Monomial(Complex(0), [])
        # Check if single element
        if len(self.summands) == 1:
            return self.summands[0].reduce()
        # Check if summands are sums themselves and unpack
        if any(isinstance(summand, Sum) for summand in self.summands):
            #return Sum(*(summand.summands if isinstance(summand, Sum) else summand for summand in self.summands)).reduce()
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
            return Sum(*(summand for summand in self.summands if not summand == Monomial(Complex(0))))
        # At this point, every summand should be a non-zero monomial
        if not all(isinstance(summand, Monomial) for summand in self.summands):
            raise Exception("In this almost reduced sum, every summand should be a monomial.")
        # Group monomials with the same signature
        # Use the signature as the key in a dict, and the coefficient as the value, adding coefficients of terms with
        # the same signature
        d = defaultdict(lambda: Complex(0))
        for summand in self.summands:
            assert isinstance(summand, Monomial)
            d[summand.signature()] += summand.coefficient
        return Sum(*(Monomial(coefficient, simple_factors)
                     for simple_factors, coefficient in d.items() if coefficient != 0)).reduce()

    def canonicalize(self):
        if not self.is_reduced:
            return self.reduce().canonicalize()
        return Sum(*map(lambda e: e.canonicalize(), self.summands)).reduce()

    def __str__(self):
        return reduce(lambda a, b: a + " + " + b, map(str, self.summands))


class Product(Element):

    def __init__(self, *factors: Element):
        self.factors: list[Element] = list(factors)
        super().__init__()

    def __add__(self, other):
        return Sum(self, other)

    @overload
    def __mul__(self, other: 'Product') -> 'Product': ...

    def __mul__(self, other):
        if isinstance(other, Product):
            return Product(*self.factors, *other.factors)
        return Product(*(self.factors + [other]))

    def __cmp__(self, other):
        raise NotImplementedError

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
        list_of_summand_lists: [[Monomial]] = [[Monomial(Rational(1), [])]]
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

    def __str__(self):
        return reduce(lambda a, b: "(" + a + ") (" + b + ")", map(str, self.factors))


basisVectors = generate_sl(3)
ad = generate_ad_action_matrix_index_to_element(basisVectors)

# Regular elements
# Ys
e21 = Monomial(Complex(1), [(0, 1)])
e31 = Monomial(Complex(1), [(2, 1)])
e32 = Monomial(Complex(1), [(3, 1)])
# Hs
h1 = Monomial(Complex(1), [(3, 1)])
h2 = Monomial(Complex(1), [(4, 1)])
# Xs
e12 = Monomial(Complex(1), [(5, 1)])
e23 = Monomial(Complex(1), [(6, 1)])
e13 = Monomial(Complex(1), [(7, 1)])


# 'Dual' elements
H1 = Monomial(Complex(Rational(1,9))) * h1 + Monomial(Complex(1/18)) * h2
H2 = Monomial(Complex(1/18)) * h1 + Monomial(Complex(1/9)) * h2
E12 = Monomial(Complex(1/6)) * e12
E23 = Monomial(Complex(1/6)) * e23
E13 = Monomial(Complex(1/6)) * e13
E21 = Monomial(Complex(1/6)) * e21
E32 = Monomial(Complex(1/6)) * e32
E31 = Monomial(Complex(1/6)) * e31


# E12 H1 = H1 E12 - [H1,E12]
# 5 3 -> 3 5 - noncommutative_part[3][5]
# noncommutative_part is like ad
minus = Monomial(Complex(-1))
noncommutative_part = [[Monomial(Complex(0)) for _ in range(len(basisVectors))]]
#noncommutative_part[0][2] = minus * e31
#noncommutative_part[0][3] = None #  Muss ich noch ausrechnen


def reduced_casimir_second_order():
    six = Monomial(Complex(6))
    casimir_2 = (six + six) * H1 * H1 - six * H1 * H2 - six * H2 * H1 +(six + six) * H2 * H2 +\
              six * (E12 * E21 + E21 * E12 + E13 * E31 + E31 * E13 + E23 * E32 + E32 * E23)
    reduced_casimir_2 = casimir_2.reduce()
    print(reduced_casimir_2)


def reduced_casimir_third_order():
    casimir_3 = Monomial(Complex(10)) * (E12*E23*E31 - E12*H1*E21 + E12*H2*E21
                                          - E23*E32*H1 + E23*E32*H2 + E23*E31*E12 - E23*H2*E32
                                          + E12*E32*E21 + E13*E31*H1 - E13*H2*E31
                                          - E21*E12*H1 + E21*E12*H2 + E21*E13*E32 + E21*H1*E12
                                          - E32*E23*H2 + E32*E21*E13 - E32*H1*E23 + E32*H2*E23
                                          + E31*E12*E23 - E31*E13*H2 + E31*H1*E13
                                          + H1*E12*E21 - H1*E23*E32 + H1*E13*E31 - H1*E21*E12 + H1*H1*H2 + H1*H2*H1 - H1*H2*H2
                                          + H2*E23*E32 + H2*E21*E12 - H2*E32*E23 - H2*E31*E13 - H2*H1*H2 + H2*H1*H1 - H2*H2*H1)
    reduced_casimir_3 = casimir_3.reduce()
    print(reduced_casimir_3)
    return reduced_casimir_3

if __name__ == "__main__":
    # ads = generate_ad_action_matrices(basisVectors)
    # k = np.array([[np.trace(ads[i] @ ads[j]) for j in range(len(basisVectors))] for i in range(len(basisVectors))])
    # e = e12 * e12 * h1
    # print(e)
    # print(e.canonicalize())
    # c = reduced_casimir_third_order()
    # cc = c.canonicalize()
    # print(cc)
    one = Complex(1, 0)
    i = Complex(0, 1)
    e = one + i*one
    b = Complex(1,1)
    print(b == e)
    print(e**2)
    print(3* e )