import os
import sys
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.plot.utils.plot3DCurve import (
        plot3DCurve,
    )
    from src.modelling.utils.calculateArcLength import (
        calcArcLengthFromCurveFun,
        calcArcLengthFromPolygonization,
    )
    from src.modelling.curveShapes3D import helixShape
except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise

vis = False  # enable for visualization


def testHelixShape():
    s = np.linspace(0, 1, 1000)
    testHelix = lambda s: helixShape(s, heightScaling=0.0, frequency=2.0)
    arcLenght_Integral = calcArcLengthFromCurveFun(testHelix, 0, 1)
    approxArcLenght_Polygon = calcArcLengthFromPolygonization(testHelix, 0, 1)

    if vis:
        testHelixPositions = testHelix(s)
        plot3DCurve(testHelixPositions, block=True)
    assert arcLenght_Integral == approx(2 * np.pi, 0.001)
    assert approxArcLenght_Polygon == approx(2 * np.pi, 0.001)


if __name__ == "__main__":
    testHelixShape()
