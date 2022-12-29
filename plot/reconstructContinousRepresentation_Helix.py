import os
import sys
import numpy as np
from functools import partial
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
    )
    from src.reconstruction.continuous.continuousReconstruction import (
        ContinuousReconstruction,
    )
except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise


def setupVisualizationCallback(continousReconstruction):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizationCallback,
        fig,
        ax,
        continousReconstruction,
        [-3, 3],
        [-3, 3],
        [0, 6],
        savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    continuousModel,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    plt.cla()
    visualizePointSets(
        X=continuousModel.X,
        Y=continuousModel.Y,
        ax=ax,
        axisLimX=axisLimX,
        axisLimY=axisLimY,
        axisLimZ=axisLimZ,
    )
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(continuousModel.iter) + ".png")


if __name__ == "__main__":

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
            "Rtor": 1000,  # use 1000
            "Rflex": 1000,  # use 1000
            "Roh": 10,
            "wPosDiff": 10,  # use 10
            #            "aPhi": aPhi,
            #            "aTheta": aTheta,
            "annealingFlex": 0.9,  # use 0.99
            "annealingTor": 0.8,  # use 0.8
        }
    )
    visCallback = setupVisualizationCallback(continousReconstruction)
    continousReconstruction.registerCallback(visCallback)
    continousReconstruction.estimateShape(numIter=None)
    continousReconstruction.writeParametersToJson(
        savePath="plot/plotdata/", fileName="helixExample_2"
    )
