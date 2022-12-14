import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.plot.utils.plot3DCurve import (
        plot3DCurve,
    )
    from src.modelling.utils.calculateArcLength import calcArcLengthFromCurveFun
    from src.modelling.curveShapes3D import helixShape
    from src.plot.utils.visualization import (
        visualizePointSets,
        setupVisualizationCallback,
    )
    from src.reconstruction.differentialGeometry.differentialGeometryReconstruction import (
        DifferentialGeometryReconstruction,
    )
except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise

if __name__ == "__main__":

    visCallback = setupVisualizationCallback([-5, 5], [-5, 5], [-5, 5])

    # helix definition & reconstuction
    helixCurve = lambda s: helixShape(s, heightScaling=1.0, frequency=2.0)
    arcLenght = calcArcLengthFromCurveFun(helixCurve, 0, 1)
    s = np.linspace(0, 1, 30)
    Sx = s * arcLenght
    Y = helixCurve(s)
    Y_rot = Y.copy()
    Y_rot[:, 2] = Y[:, 1]
    Y_rot[:, 1] = -Y[:, 2]
    continousReconstruction = DifferentialGeometryReconstruction(
        **{
            "Y": Y_rot,
            "Sx": Sx,
            "L": arcLenght,
            "numSc": 10,
            "callback": visCallback,
            "Rtor": 10,
            "Rflex": 0.01,
            "Density": 0.1,
            "wPosDiff": 0.3,
        }
    )
    continousReconstruction.estimateShape(numIter=100)
