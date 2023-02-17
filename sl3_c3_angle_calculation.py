import sympy
from sympy import cos, sin, symbols
from sympy import sqrt as ssqrt
from numpy import sqrt
from numpy import arctan as atan

if __name__ == "__main__":
    a, t = symbols('a t')
    x = 2 * ssqrt(a + 3) * cos(t)
    y = 2 * ssqrt((a + 3) / 3) * sin(t)
    print("x:", x)
    print("y:", y)
    q = (x + y - 2) / 2
    p = (y - x - 2) / 2
    print("q:", q)
    print("p:", p)
    c2 = q ** 2 + q * p + p ** 2 + 3 * q + 3 * p
    print("a:", c2.simplify())
    c3 = 2 * q ** 3 + 3 * q ** 2 * p - 3 * q * p ** 2 - 2 * p ** 3 + 9 * q ** 2 - 9 * p ** 2 + 9 * q - 9 * p
    hight_function = c3.simplify()
    phi_min = sympy.solve(p, t)[0]
    phi_max = sympy.solve(q, t)[1]
    # sympy.limit(phi_max, a, sympy.oo).doit()  # Does not work, use continuity of atan or wolframalpha or pen and paper