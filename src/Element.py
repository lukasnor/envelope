import abc
import numbers
from typing import Hashable, TYPE_CHECKING

from sympy import Rational

from src.Complex import Complex

if TYPE_CHECKING:
    from src.BasisVector import BasisVector
Replacement = dict['BasisVector', 'Element']

class Element(abc.ABC):

    def __init__(self):
        self.is_reduced = self._determine_reduced()

    def __add__(self, other: 'Element') -> 'Sum':
        from src.Sum import Sum
        return Sum(self, other)

    def __neg__(self):
        return Complex(-1) * self

    def __sub__(self, other: 'Element') -> 'Sum':
        return self + - other

    def __mul__(self, other: 'Element') -> 'Product':
        from src.Product import Product
        return Product(self, other)

    def __rmul__(self, other):
        from src.Monomial import Monomial
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
        from src.Product import Product
        return Product(*(self for _ in range(power)))

    @abc.abstractmethod
    def __str__(self) -> str:
        ...

    @abc.abstractmethod
    def signature(self) -> Hashable:
        ...

    @abc.abstractmethod
    def _determine_reduced(self) -> bool:
        ...

    @abc.abstractmethod
    def reduce(self) -> 'Element':
        ...

    @abc.abstractmethod
    def canonicalize(self) -> 'Element':
        ...

    @abc.abstractmethod
    def replace(self, replacement: Replacement) -> 'Element':
        ...

    def normalize(self) -> 'Element': # Formally "-> Self" but not in Python 3.10 yet
        return self

    def sort_by_degree(self) -> 'Element':  # Formally "-> Self" but not in Python 3.10 yet
        return self

    def group_by_coefficient(self) -> 'Element': # Formally "-> Self" but not in Python 3.10 yet
        return self
