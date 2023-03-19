import os
import sys
import numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.bdloReconstruction import (
        BDLOReconstruction,
    )
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.simulation.bdloTemplates import initArenaWireHarness
    from src.visualization.plot3D import *
except:
    print("Imports for BDLOReconstruction Test failed.")
    raise


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
    ax.cla()
    plotPointSets(
        ax=ax,
        X=classHandle.X,
        Y=classHandle.Y,
        ySize=10,
        xSize=10,
    )
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def runReconstruction():
    testPointSet = np.loadtxt(dataPath)
    testBDLO = initArenaWireHarness()
    testCorrespondanceEstimator = TopologyBasedCorrespondanceEstimation(
        **{
            "Y": testPointSet,
            "extractedTopology": MinimalSpanningTreeTopology(testPointSet),
            "numSeedPoints": 70,
            "templateTopology": testBDLO,
        }
    )
    Y = testPointSet
    (
        CBY,
        SY,
    ) = testCorrespondanceEstimator.calculateBranchCorresponanceAndLocalCoordinatsForPointSet(
        Y
    )
    testReconstruction = BDLOReconstruction(
        **{"bdlo": testBDLO, "Y": Y, "SY": SY, "CBY": CBY}
    )

    if vis:
        visualizationCallback = setupVisualizationCallback(testReconstruction)
        testReconstruction.registerCallback(visualizationCallback)

    testReconstruction.reconstructShape()


if __name__ == "__main__":
    runReconstruction()
