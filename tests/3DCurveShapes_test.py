import os
import sys
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.visualization.plot3DCurve import (
        plot3DCurve,
    )
    from src.modelling.utils.calculateArcLength import (
        calcArcLengthFromCurveFun,
        calcArcLengthFromPolygonization,
    )
    from src.visualization.curveShapes3D import helixShape, gaussianShape
except:
    print("Imports for 3D Curve Shape Tests failed.")
    raise

vis = True  # enable for visualization


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


def testGaussianShape():
    s = np.linspace(0, 1, 100)
    testGassian = lambda s: gaussianShape(
        s, mu=0.5, width=0.02, height=0.4, offset=np.array([0, 0.5, 0])
    )
    if vis:
        plot3DCurve(
            testGassian(s),
            axisLimX=[0, 1],
            axisLimY=[0, 1],
            axisLimZ=[0, 1],
            block=True,
        )


if __name__ == "__main__":
    # testHelixShape()
    testGaussianShape()
