import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.discrete.discreteReconstruction import (
        DiscreteReconstruction,
    )
    from plot.utils.visualization import (
        visualizePointSets,
    )
except:
    print("Imports for DiscreteReconstruction Test failed.")
    raise
vis = True  # enable for visualization
visAtEnd = True


def setupVisualizationCallback(discreteReconstruction):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizationCallback,
        fig,
        ax,
        discreteReconstruction,
        [-10, 100],
        [-10, 100],
        [-10, 100],
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    discreteModel,
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
        X=discreteModel.X,
        Y=discreteModel.Y,
        ax=ax,
        axisLimX=axisLimX,
        axisLimY=axisLimY,
        axisLimZ=axisLimZ,
    )
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(discreteModel.iter) + ".png")


def runReconstruction():
    length = 100
    numDisc = 10
    Y = np.ones((numDisc, 3))
    Y[:, 1] = np.linspace(0, length, numDisc)
    # Y[:, 2] = np.linspace(0, 50, 10)
    # X[5, 2] = 55
    SY = np.linspace(0, length, numDisc) / length
    testReconstruction = DiscreteReconstruction(
        **{
            "Y": Y,
            "SY": SY,
            "L": length,
            "x0": Y[0, :],
            "N": 10,
            "wPosDiff": 100,
            "Rflex": 1,
            "Rtor": 1,
            "Roh": 1,
        }
    )
    if vis:
        visCallback = setupVisualizationCallback(testReconstruction)
        testReconstruction.registerCallback(visCallback)

    testReconstruction.reconstructShape(numIter=100)

    if visAtEnd:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        visualizePointSets(
            X=testReconstruction.X,
            Y=testReconstruction.Y,
            ax=ax,
            axisLimX=[-10, 100],
            axisLimY=[-10, 100],
            axisLimZ=[-10, 100],
            waitTime=-1,
        )


if __name__ == "__main__":
    runReconstruction()
