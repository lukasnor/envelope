from collections import Counter, defaultdict
from functools import reduce
from typing import overload, Dict, Tuple

from src.Complex import Complex
from src.Element import Element


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
        from src.Monomial import Monomial
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
        from src.Monomial import Monomial
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
        from src.BasisVector import BasisVector
        d: Dict[Tuple[Tuple[BasisVector, int]], Complex] = defaultdict(lambda: Complex(0))
        for summand in self.summands:
            assert isinstance(summand, Monomial)
            d[summand.signature()] += summand.coefficient
        return Sum(*(Monomial(coefficient, simple_factors) for simple_factors, coefficient in d.items())).reduce()

    def canonicalize(self):
        if not self.is_reduced:
            return self.reduce().canonicalize()
        return Sum(*map(lambda e: e.canonicalize(), self.summands)).reduce()