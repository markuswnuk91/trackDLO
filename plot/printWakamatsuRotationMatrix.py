from sympy import *


def Rx(alpha):
    return Matrix(
        [[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]]
    )


def Ry(beta):
    return Matrix([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])


def Rz(gamma):
    return Matrix(
        [[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]]
    )


init_printing(use_unicode=True)
phi, theta, psi = symbols("phi theta psi")

Rxyz_extrinsic = Rz(phi) * Ry(theta) * Rz(psi)
pprint(Rxyz_extrinsic)
# print(latex(Rxyz_extrinsic))

# Rxyz_intrinsic = Rx * Ry * Rz
# pprint(Rxyz_intrinsic)
