import copy
from functools import reduce
from typing import overload

from src.Complex import Complex
from src.Element import Element, Replacement


class Product(Element):

    def __init__(self, *factors: Element):
        self.factors: list[Element] = list(factors)
        super().__init__()

    @overload
    def __mul__(self, other: 'Product') -> 'Product':
        ...

    def __mul__(self, other: Element):
        if isinstance(other, Product):
            return Product(*self.factors, *other.factors)
        elif isinstance(other, Element):
            return Product(*(self.factors + [other]))
        else:
            return NotImplemented

    def __str__(self):
        return reduce(lambda a, b: "(" + a + ") (" + b + ")", map(str, self.factors))

    def signature(self):
        return tuple(factor.signature() for factor in self.factors)

    def _determine_reduced(self):
        return False

    def reduce(self):
        from src.Monomial import Monomial
        from src.Sum import Sum
        if self.is_reduced:
            return self
        # Check if empty product
        if len(self.factors) == 0:
            return Monomial(Complex(1), [])
        # Check if only one factor
        if len(self.factors) == 1:
            return self.factors[0].reduce()
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
        # If all factors are monomials, no distribution must happen
        if all(isinstance(factor, Monomial) for factor in self.factors):
            return reduce(Monomial.__mul__, self.factors)
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
        # return Product(*map(lambda e: e.canonicalize(), self.factors)).reduce()

    @overload
    def replace(self, replacement: Replacement) -> 'Product':
        ...

    def replace(self, replacement: Replacement) -> Element:
        return Product(*(factor.replace(replacement) for factor in self.factors))
