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
    from src.visualization.plot3D import *
except:
    print("Imports for Topology Extraction Test failed.")
    raise

# control parameters
dataPath = "tests/testdata/topologyExtraction/wireHarness.txt"
vis = True  # enable for visualization


def setupVisualization(dim):
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    elif dim <= 2:
        fig = plt.figure()
        ax = fig.add_subplot()
    # set axis limits
    # ax.set_xlim(0.2, 0.8)
    # ax.set_ylim(-0.3, 0.3)
    # ax.set_zlim(0, 0.6)
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
        X=classHandle.T,
        Y=classHandle.Y,
        ySize=5,
        xSize=10,
        # yMarkerStyle=".",
        yAlpha=0.03,
    )
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)
    print(classHandle.iteration)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def test_topologyExtraction():
    testPointSet = np.loadtxt(dataPath)
    somParameters = {
        "alpha": 1,
        "numNearestNeighbors": 30,
        "numNearestNeighborsAnnealing": 0.8,
        "sigma2": 0.03,
        "alphaAnnealing": 0.9,
        "sigma2Annealing": 0.8,
        "kernelMethod": False,
        "max_iterations": 30,
    }

    l1Parameters = {
        "h": 0.12,
        "hAnnealing": 0.8,
        "hReductionFactor": 1,
        "mu": 0.35,
        "max_iterations": 30,
    }
    lofOutlierFilterParameters = {
        "numNeighbors": 15,
        "contamination": 0.1,
    }
    testTopologyExtractor = TopologyExtraction(
        **{
            "Y": testPointSet,
            "numSeedPoints": 100,
            "somParameters": somParameters,
            "l1Parameters": l1Parameters,
            "lofOutlierFilterParameters": lofOutlierFilterParameters,
        }
    )
    if vis:
        somVisualizationCallback = setupVisualizationCallback(
            testTopologyExtractor.selfOrganizingMap
        )
        testTopologyExtractor.selfOrganizingMap.registerCallback(
            somVisualizationCallback
        )

        l1VisualizationCallback = setupVisualizationCallback(
            testTopologyExtractor.l1Median
        )
        testTopologyExtractor.l1Median.registerCallback(l1VisualizationCallback)

    testTopology = testTopologyExtractor.extractTopology(numSeedPoints=70)

    if vis:
        fig, ax = setupVisualization(testTopology.X.shape[1])
        pointPairs = testTopology.getAdjacentPointPairs()
        leafNodeIndices = testTopology.getLeafNodeIndices()
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
        plotPointSet(ax=ax, X=testTopology.X, color=[1, 0, 0], size=30)
        plotPointSet(ax=ax, X=testTopology.X, color=[1, 0, 0], size=20)
        plotPointSet(
            ax=ax,
            X=testTopology.X[leafNodeIndices, :],
            color=[1, 0, 0],
            size=50,
            alpha=0.4,
        )
        set_axes_equal(ax)
        plt.show(block=True)


if __name__ == "__main__":
    test_topologyExtraction()
