import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.visualization.plot3D import plotPointSets
except:
    print("Imports for Topology Extraction Test failed.")
    raise

dataPath = "tests/testdata/topologyExtraction/topologyExtractionTestSet.txt"

vis = True  # enable for visualization

# control parameters


def setupVisualizationCallback(classHandle):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizationCallback,
        fig,
        ax,
        classHandle,
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    classHandle,
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    plt.cla()
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.6)
    plotPointSets(
        ax=ax,
        X=classHandle.T,
        Y=classHandle.Y,
        ySize=5,
        xSize=10,
        # yMarkerStyle=".",
        yAlpha=0.01,
        waitTime=0.5,
    )
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def test_topologyExtraction():
    testPointSet = np.loadtxt(dataPath)

    testTopologyExtractor = TopologyExtraction(
        **{
            "X": testPointSet,
        }
    )
    testMinSpanTree = testTopologyExtractor.extractTopologyRepresentation()

    # visualization
    if vis:
        # plot initial point set
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            testPointSet[:, 0],
            testPointSet[:, 1],
            testPointSet[:, 2],
        )
        plt.show(block=True)


if __name__ == "__main__":
    test_topologyExtraction()
