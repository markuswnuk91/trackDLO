import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.continuousReconstruction import (
        ContinuousReconstruction,
    )
    from src.visualization.plot3D import (
        plotPointSets,
    )
except:
    print("Imports for ContinuousReconstruction failed.")
    raise
vis = True  # enable for visualization


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
    Y = np.ones((10, 3))
    Y[:, 1] = np.linspace(0, 100, 10)
    # Y[:, 2] = np.linspace(0, 50, 10)
    # X[5, 2] = 55
    SY = np.linspace(0, 100, 10) / 100
    if vis:
        testReconstruction = ContinuousReconstruction(
            **{
                "Y": Y,
                "SY": SY,
                "L": 100,
                "numSc": 100,
                "Rflex": 1,
                "Rtor": 1,
                "Roh": 0,
            }
        )
        visCallback = setupVisualizationCallback(testReconstruction)
        testReconstruction.registerCallback(visCallback)
    else:
        testReconstruction = ContinuousReconstruction(
            **{"Y": Y, "SY": SY, "L": 100, "numSc": 100}
        )
    testReconstruction.reconstructShape()

    testReconstruction


if __name__ == "__main__":
    runReconstruction()
