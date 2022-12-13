import numpy as np
import scipy.integrate as integrate
from scipy.misc import derivative


def calcArcLengthFromUnitTangent(unitTangentVectorFun, s_low, s_up):
    """calcualtes the arclength of a 3D curve

    Args:
        unitTangentVectorFun (function object): function f'(s) returning the tangent unit vector of the 3D curve.
        s_low (float): lower bound of the integration
        s_up (float): upper bound of the integration

    Returns:
        float: arclenth of the curve between the upper and the lower bound.
    """
    return integrate.quad(
        lambda s: np.linalg.norm(unitTangentVectorFun(s)), s_low, s_up
    )


def calcArcLengthFromCurveFun(curveFun, s_low, s_up, ds=1e-8):
    """calcualtes the arclength of a 3D curve

    Args:
        curveFun (function object): function f(s) returning a NxD position vector of the 3D curve for every s, where N is the number of local coodinates and D are the dimension.
        s_low (float): lower bound of the integration
        s_up (float): upper bound of the integration
        ds (flaot): spacing along s for the approximation of the cuves tangent vector
    Returns:
        float: arclenth of the curve between the upper and the lower bound.
    """
    return integrate.quad(
        lambda s: np.linalg.norm(derivative(curveFun, s, ds)), s_low, s_up
    )[0]


def calcArcLengthFromPolygonization(curveFun, s_low, s_up, numPoints=1000):
    s = np.linspace(s_low, s_up, numPoints)
    curveFunValues = curveFun(s)
    linDiffs = np.diff(curveFunValues, axis=0)
    return np.sum(np.linalg.norm(linDiffs, axis=1))
