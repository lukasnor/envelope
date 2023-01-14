import functools
from typing import Tuple

import numpy as np

from src.Complex import Complex
from src.Element import Element
from src.Monomial import Monomial


@functools.total_ordering  # Lazy, but computationally costly, implementation of le, ge, lt, gt, ne, eq
class BasisVector:

    # Each basis vector is determined by its index. Equality '==' and hashing depends on the index alone
    # The ad action has to be defined "manually" in post, once a complete basis is constructed,
    # for a canonicalization to work
    # The matrix argument should be used to give a representation to the elements
    # TODO: Abstract away the representation
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

    # TODO: Decide if preferred functionality or just hacky. Why not inherit from Element alltogether?
    # The only sensible way to add to a BasisVector is to convert it to an element and try again
    def __add__(self, other):
        return Monomial(Complex(1), [(self, 1)]) + other

    def __radd__(self, other):
        return other + Monomial(Complex(1), [(self, 1)])

    # The only sensible way to multiply to a BasisVector is to convert it to an element and try again
    def __mul__(self, other):
        return Monomial(Complex(1), [(self, 1)]) * other

    def __rmul__(self, other):
        return other * Monomial(Complex(1), [(self, 1)])

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
