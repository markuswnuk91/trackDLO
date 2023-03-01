import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.discreteReconstruction import (
        DiscreteReconstruction,
    )
    from src.visualization.plot3D import (
        plotPointSets,
    )
except:
    print("Imports for DiscreteReconstruction Test failed.")
    raise
vis = False  # enable for visualization
visAtEnd = True


def setupVisualizationCallback(reconstruction):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizationCallback,
        fig,
        ax,
        reconstruction,
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    model,
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    plt.cla()

    # set axis limits
    ax.set_xlim(-10, 100)
    ax.set_ylim(-10, 100)
    ax.set_zlim(-10, 100)

    plotPointSets(
        X=model.X,
        Y=model.Y,
        ax=ax,
    )

    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(model.iter) + ".png")


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
            "Rflex": 0,
            "Rtor": 0,
            "Roh": 1,
        }
    )
    if vis:
        visCallback = setupVisualizationCallback(testReconstruction)
        testReconstruction.registerCallback(visCallback)

    testReconstruction.reconstructShape(numIter=500)

    if visAtEnd:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # set axis limits
        ax.set_xlim(-10, 100)
        ax.set_ylim(-10, 100)
        ax.set_zlim(-10, 100)
        plotPointSets(
            X=testReconstruction.X,
            Y=testReconstruction.Y,
            ax=ax,
            waitTime=None,
        )


if __name__ == "__main__":
    runReconstruction()
