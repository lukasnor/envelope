import abc
import numbers
from typing import Hashable

from sympy import Rational

from src.Complex import Complex
from src.Monomial import Monomial
from src.Product import Product
from src.Sum import Sum


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

    # @abstractmethod
    # def replace(self) -> 'Element': ...
