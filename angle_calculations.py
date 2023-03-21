import sympy
from sympy import symbols, sqrt, sin, cos

if __name__ == "__main__":
    n, A, phi = symbols("n A phi")
    # a = sqrt(2*n/(n-1) * (A + n*(n+1)/2))
    # b = sqrt(2*n/(n+1) * (A + n*(n+1)/2))
    a, b = symbols("a b")
    print("First solution", *sympy.solve(a*cos(phi) + b * sin(phi) - n, phi), sep="\n")
    print("Second solution", *sympy.solve(b*sin(phi) - a * cos(phi) - n, phi), sep="\n")