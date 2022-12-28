import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from plot.utils.plot3DCurve import (
        plot3DCurve,
    )
    from src.modelling.utils.calculateArcLength import calcArcLengthFromCurveFun
    from src.modelling.curveShapes3D import helixShape
    from plot.utils.visualization import (
        visualizePointSets,
        setupVisualizationCallback,
    )
    from src.reconstruction.continuous.continuousReconstruction import (
        ContinuousReconstruction,
    )
except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise


if __name__ == "__main__":

    visCallback = setupVisualizationCallback(
        [-3, 3],
        [-3, 3],
        [0, 6],
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail2/",
    )

    # helix definition & reconstuction
    helixCurve = lambda s: helixShape(s, heightScaling=1.0, frequency=2.0)
    arcLenght = calcArcLengthFromCurveFun(helixCurve, 0, 1)
    s = np.linspace(0, 1, 30)
    Y = helixCurve(s)
    continousReconstruction = ContinuousReconstruction(
        **{
            "Y": Y,
            "SY": s,
            "x0": Y[0, :],
            "L": arcLenght,
            "numSc": 30,
            "callback": visCallback,
            "Rtor": 1000,  # use 1000
            "Rflex": 1000,  # use 1000
            "Roh": 10,
            "wPosDiff": 10,  # use 10
            #            "aPhi": aPhi,
            #            "aTheta": aTheta,
            "annealingFlex": 0.99,  # use 0.99
            "annealingTor": 0.8,  # use 0.8
        }
    )
    continousReconstruction.estimateShape(numIter=None)
    continousReconstruction.writeParametersToJson(
        savePath="src/plot/plotdata/", fileName="helixExample_2"
    )
