import numbers
from functools import reduce

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

    def __hash__(self):
        return (self.re.numerator, self.re.denominator, self.im.numerator, self.im.denominator).__hash__()

    def __abs__(self):
        return Complex(sqrt(self.re ** 2 + self.im ** 2), 0)

    def real(self):
        return Complex(self.re, 0)

    def imag(self):
        return Complex(self.im, 0)

    def conjugate(self):
        return Complex(self.re, - self.im)

if __name__ == "__main__":
    one = Complex(1, 0)
    i = Complex(0, 1)
    e = one + i*one
    b = Complex(1, 1)
    print(b == e)
    print(e**2)
    print(3 * e)
    i.__hash__()
