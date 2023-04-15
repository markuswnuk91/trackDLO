import os
import sys
import numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.localization.bdloLocalization import (
        BDLOLocalization,
    )
    from src.simulation.bdloTemplates import initArenaWireHarness
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.visualization.plot3D import *
except:
    print("Imports for BDLOReconstruction Test failed.")
    raise

# script control parameters
vis = True  # enable for visualization
dataPath = "tests/testdata/topologyExtraction/wireHarnessReduced.txt"


def setupVisualization(dim):
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    elif dim <= 2:
        fig = plt.figure()
        ax = fig.add_subplot()
    return fig, ax


def setupVisualizationCallback(classHandle):
    fig, ax = setupVisualization(classHandle.Y.shape[1])
    return partial(
        visualizationCallback,
        fig,
        ax,
        classHandle,
        savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/bldoReconstruction/test/",
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
    ax.cla()
    plotPointSets(
        ax=ax,
        X=classHandle.X,
        Y=classHandle.YTarget,
        ySize=10,
        xSize=10,
    )
    for i, y in enumerate(classHandle.YTarget):
        plotLine(
            ax=ax,
            pointPair=np.vstack(((classHandle.C @ classHandle.X)[i], y)),
            color=[1, 0, 0],
            alpha=0.3,
        )
    plt.draw()
    plt.pause(0.1)


def runReconstruction():
    testPointSet = np.loadtxt(dataPath)
    templateTopologyModel = initArenaWireHarness()
    extractedTopologyModel = MinimalSpanningTreeTopology(testPointSet)
    testPointSet = np.loadtxt(dataPath)

    Y = testPointSet
    S = np.linspace(0, 1, 10)
    testLocalization = BDLOLocalization(
        **{
            "Y": Y,
            "S": S,
            "templateTopology": templateTopologyModel,
            "extractedTopology": extractedTopologyModel,
        }
    )

    if vis:
        visualizationCallback = setupVisualizationCallback(testLocalization)
        testLocalization.registerCallback(visualizationCallback)

    testLocalization.reconstructShape()


if __name__ == "__main__":
    runReconstruction()
