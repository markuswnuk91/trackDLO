import os
import sys
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.modelling.utils.calculateArcLength import calcArcLengthFromCurveFun
    from src.visualization.curveShapes3D import helixShape
    from src.visualization.plot3D import (
        plotPointSets,
        plotPointSetAsLine,
        plotPointSetsAsLine,
    )
    from src.reconstruction.continuousReconstruction import ContinuousReconstruction
    from src.reconstruction.discreteReconstruction import DiscreteReconstruction
    from src.modelling.utils.utils import (
        loadWakamatsuModelParametersFromJson,
    )

except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise

saveParams = False
plotSteps = False
plotFinal = True

# continuous reconstruction
reconstructContinuous = False
numIterContinuous = None
reconstructFromScratch = False
loadPathInitialParameters = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/plot/plotdata/helixReconstruction/helix_continuousModel.json"

# discrete reconsruction
reconstructDiscrete = True
numIterDiscrete = 300
numSegments = 5

savePath = "plot/plotdata/helixReconstruction/"
fileName_continuousParams = "helix_continuousModel"


def reconstructParams():
    return {
        "numSc": 30,
        "Rtor": 10,  # use 1000
        "Rflex": 10,  # use 1000
        "Roh": 0,
        "wPosDiff": 10,  # use 10
        #            "aPhi": aPhi,
        #            "aTheta": aTheta,
        "annealingFlex": 1,  # use 0.995
        "annealingTor": 1,  # use 0.8
    }


def setupVisualizationCallback(shapeReconstruction):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizationCallback,
        fig,
        ax,
        shapeReconstruction,
        [-3, 3],
        [-3, 3],
        [0, 6],
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    reconstructedModel,
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
    # set axis limits
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    plotPointSets(
        X=reconstructedModel.X,
        Y=reconstructedModel.Y,
        ax=ax,
    )

    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(reconstructedModel.iter) + ".png")


def reconstructFromScratchParams():
    # do not change
    return {
        "numSc": 30,
        "Rtor": 1000,  # use 1000
        "Rflex": 1000,  # use 1000
        "Roh": 0,
        "wPosDiff": 10,  # use 10
        #            "aPhi": aPhi,
        #            "aTheta": aTheta,
        "annealingFlex": 0.995,  # use 0.995
        "annealingTor": 0.8,  # use 0.8
    }


if __name__ == "__main__":

    # helix definition & reconstuction
    helixCurve = lambda s: helixShape(s, heightScaling=2.0, frequency=2)
    arcLength = calcArcLengthFromCurveFun(helixCurve, 0, 1)

    s = np.linspace(0, 1, 30)
    Y = helixCurve(s)

    sDiscrete = np.linspace(0, 1, numSegments + 1)
    discreteY = helixCurve(sDiscrete)

    # reconstruct with contiuous model
    if reconstructFromScratch:
        reconstructionParams = reconstructFromScratchParams()
        continousReconstruction = ContinuousReconstruction(
            **{
                "Y": Y,
                "SY": s,
                "x0": Y[0, :],
                "L": arcLength,
                "numSc": reconstructionParams["numSc"],
                "Rtor": reconstructionParams["Rtor"],
                "Rflex": reconstructionParams["Rflex"],
                "Roh": reconstructionParams["Roh"],
                "wPosDiff": reconstructionParams["wPosDiff"],
                "annealingFlex": reconstructionParams["annealingFlex"],
                "annealingTor": reconstructionParams["annealingTor"],
            }
        )
    else:
        reconstructionParams = reconstructParams()
        modelParams = loadWakamatsuModelParametersFromJson(loadPathInitialParameters)
        continousReconstruction = ContinuousReconstruction(
            **{
                "Y": Y,
                "SY": s,
                "x0": Y[0, :],
                "L": arcLength,
                "aPhi": modelParams["aPhi"],
                "aTheta": modelParams["aTheta"],
                "aPsi": modelParams["aPsi"],
                "numSc": reconstructionParams["numSc"],
                "Rtor": reconstructionParams["Rtor"],
                "Rflex": reconstructionParams["Rflex"],
                "Roh": reconstructionParams["Roh"],
                "wPosDiff": reconstructionParams["wPosDiff"],
                "annealingFlex": reconstructionParams["annealingFlex"],
                "annealingTor": reconstructionParams["annealingTor"],
            }
        )

    correspondanceWeightingFactor = np.ones(numSegments + 1)
    correspondanceWeightingFactor[2] = 100

    # correspondanceWeightingFactor[-1] = 1
    discreteReconstruction = DiscreteReconstruction(
        **{
            "Y": discreteY,
            "SY": sDiscrete,
            "x0": Y[0, :],
            "L": arcLength,
            "N": numSegments,
            "correspondanceWeightingFactor": correspondanceWeightingFactor,
        }
    )

    # run reconstructions
    if reconstructContinuous:
        if plotSteps:
            # continuous
            visCallbackContinuousModel = setupVisualizationCallback(
                continousReconstruction
            )
            continousReconstruction.registerCallback(visCallbackContinuousModel)
        continousReconstruction.reconstructShape(numIter=numIterContinuous)

    if reconstructDiscrete:
        if plotSteps:
            # discrete
            visCallbackDiscreteModel = setupVisualizationCallback(
                discreteReconstruction
            )
            discreteReconstruction.registerCallback(visCallbackDiscreteModel)
        discreteReconstruction.reconstructShape(numIter=numIterDiscrete)

    if reconstructContinuous and saveParams:
        continousReconstruction.writeParametersToJson(
            savePath=savePath, fileName=fileName_continuousParams
        )

    if reconstructContinuous and plotFinal:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # set axis limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-0, 6)
        plotPointSets(
            X=continousReconstruction.X,
            Y=continousReconstruction.Y,
            ax=ax,
        )

    if reconstructDiscrete and plotFinal:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # set axis limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-0, 6)
        sJoint = discreteReconstruction.getJointLocalCoordinates()
        XJoint = discreteReconstruction.getCaresianPositionsFromLocalCoordinates(sJoint)
        YStartEnd = Y[[0, -1], :]
        plotPointSets(
            X=XJoint,
            Y=YStartEnd,
            ax=ax,
        )
        plotPointSetsAsLine(
            ax=ax,
            X=discreteReconstruction.getCaresianPositionsFromLocalCoordinates(sJoint),
            Y=discreteReconstruction.Y,
        )
