from typing import Iterable, Tuple, overload

from src.BasisVector import BasisVector
from src.Complex import Complex
from src.Element import Element
from src.Product import Product
from src.Sum import Sum


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
        current_vector: 'BasisVector' = self.simple_factors[0][0]
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
        for w in range(len(self.simple_factors) - 1):
            # Here the ordering of BasisVector via the index is relevant
            # The latter comparison should be between BasisVector
            if not_determined and self.simple_factors[w][0] > self.simple_factors[w + 1][0]:
                factor_index, not_determined = w, False
                break  # an index has been found
        if not_determined:
            return self
        new_factors: ['Element'] = [Monomial(self.coefficient, self.simple_factors[:factor_index])]
        # v^n w^k = w v^n w^k-1 - (\sum_l=0^n v^l ad_w(v) v^n-1-l) w^k-1
        v, n = self.simple_factors[factor_index]
        w, k = self.simple_factors[factor_index + 1]
        new_factor = Monomial(Complex(1), [(w, 1), (v, n), (w, k - 1)]).reduce()
        summands = [Monomial(Complex(1), [(v, l)]) * w.ad(v) * Monomial(Complex(1), [(v, n - 1 - l)]) for l in range(n)]
        sum_expression = Sum(*summands).reduce()
        new_factor -= sum_expression * Monomial(Complex(1), [(w, k - 1)])
        new_factor = new_factor.reduce()
        new_factors.append(new_factor)
        new_factors.append(Monomial(Complex(1), self.simple_factors[factor_index + 2:]))
        element = Product(*new_factors).reduce()
        return element.canonicalize().reduce()

    def degree(self):
        return sum(exponent for _, exponent in self.simple_factors)
