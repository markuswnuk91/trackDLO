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

saveImgs = True
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
        Y=classHandle.Y,
        ySize=10,
        xSize=10,
    )
    for i, y in enumerate(classHandle.Y):
        plotLine(
            ax=ax,
            pointPair=np.vstack((classHandle.X[i], y)),
            color=[1, 0, 0],
            alpha=0.3,
        )

    bdloConnectedEdges = classHandle.bdlo.getAdjacentPointPairsAndBranchCorrespondance()
    for pointPair in bdloConnectedEdges:
        stackedPair = np.stack(pointPair[:2])
        branchNumber = pointPair[2]
        # plotColor = [
        #     sm.to_rgba(branchNumber)[0],
        #     sm.to_rgba(branchNumber)[1],
        #     sm.to_rgba(branchNumber)[2],
        # ]
        plotLine(ax=ax, pointPair=stackedPair, color=[0, 0, 1])
    ax.view_init(elev=40, azim=-65)
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)
    print(classHandle.iter)
    if saveImgs and savePath is not None:
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
    ) = testCorrespondanceEstimator.calculateTemplateBranchCorrespondanceAndLocalCoordinatsFromPointSet(
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
