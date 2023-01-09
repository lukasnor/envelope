from sympy import *
from sympy.physics.quantum.operator import DifferentialOperator, Derivative
from sympy.physics.quantum.state import Wavefunction
from sympy.physics.quantum.qapply import qapply

if __name__ == "__main__":
    x , y, z = symbols("x y z")
    f = Function("f")
    dx = DifferentialOperator(x*Derivative(f(x, y), y), f(x, y))
    dy = DifferentialOperator(y*Derivative(f(x, y), x), f(x, y))
    dz = DifferentialOperator(z * Derivative(f(x, y, z), z), f(x, y, z))
    w = Wavefunction(x**2*y, x, y)
    qapply(dx*dy*w)  # Wavefunction(4 * x**2 * y, x, y)