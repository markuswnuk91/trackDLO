from sympy import *

init_printing(use_unicode=True)
alpha, beta, gamma = symbols("alpha beta gamma")
Rx = Matrix([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])
pprint(Rx)
Ry = Matrix([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
pprint(Ry)
Rz = Matrix([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
pprint(Rz)

Rxyz_extrinsic = Rz * Ry * Rx
pprint(Rxyz_extrinsic)

# Rxyz_intrinsic = Rx * Ry * Rz
# pprint(Rxyz_intrinsic)
