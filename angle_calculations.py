import sympy
from sympy import symbols, sqrt, sin, cos, atan, N, oo, diff

if __name__ == "__main__":
    n, phi, p, q, A2, a, b = symbols("n phi p q A_2 a b")
    # print("First solution", *sympy.solve(a*cos(phi) + b * sin(phi) - n, phi), sep="\n")
    # print("Second solution", *sympy.solve(b*sin(phi) - a * cos(phi) - n, phi), sep="\n")
    Omega2 = (n - 1) / (2 * n) * (q - p) ** 2 \
             + (n + 1) / (2 * n) * (q + p + n) ** 2 \
             - n * (n + 1) / 2
    Omega3 = q ** 3 + 3 * q ** 2 * p / n - 3 * q * p ** 2 / n - p ** 3 \
             + 3 * (n + 1) * (q ** 2 - p ** 2) / 2 \
             + (n + 1) ** 2 * (q - p) / 2

    an = sqrt(2 * n * (A2 + n * (n + 1) / 2) / (n - 1))
    bn = sqrt(2 * n * (A2 + n * (n + 1) / 2) / (n + 1))
    x = an * cos(phi)
    y = bn * sin(phi)

    E1 = (bn + sqrt(an ** 2 + bn ** 2 - n ** 2)) / (an + n)
    E2 = (bn - sqrt(an ** 2 + bn ** 2 - n ** 2)) / (an - n)

    phi_max = 2 * atan(E1.limit(A2, oo))
    phi_min = -2 * atan(E2.limit(A2, oo))

    Omega3_in_angles = Omega3.subs(q, (x+y-n)/2).subs(p, (y-x-n)/2).simplify()