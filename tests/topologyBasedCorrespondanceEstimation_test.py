import os, sys
import numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.simulation.bdlo import BranchedDeformableLinearObject
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.localization.downsampling.l1median.l1Median import L1Median
    from src.simulation.bdloTemplates import initArenaWireHarness
    from src.visualization.plot3D import *
except:
    print("Imports for Topology Extraction Test failed.")
    raise

# control parameters
saveImgs = True
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
        savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/topologyExtraction/test",
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
    if saveImgs and savePath is not None:
        if type(classHandle) == SelfOrganizingMap:
            fig.savefig(
                savePath + fileName + "_" + "som_" + str(classHandle.iteration) + ".png"
            )
        else:
            fig.savefig(
                savePath + fileName + "_" + "l1_" + str(classHandle.iteration) + ".png"
            )


def test_topologyExtraction():
    dataPath = "tests/testdata/topologyExtraction/wireHarness.txt"
    testPointSet = np.loadtxt(dataPath)
    testBDLO = initArenaWireHarness()
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
        "h": 0.1,
        "hAnnealing": 0.93,
        "hReductionFactor": 1,
        "mu": 0.35,
        "max_iterations": 30,
    }
    lofOutlierFilterParameters = {
        "numNeighbors": 15,
        "contamination": 0.1,
    }
    testCorrespondanceEstimator = TopologyBasedCorrespondanceEstimation(
        **{
            "Y": testPointSet,
            "numSeedPoints": 70,
            "templateTopology": testBDLO,
            "somParameters": somParameters,
            "l1Parameters": l1Parameters,
            "lofOutlierFilterParameters": lofOutlierFilterParameters,
        }
    )
    if vis:
        somVisualizationCallback = setupVisualizationCallback(
            testCorrespondanceEstimator.topologyExtractor.selfOrganizingMap
        )
        testCorrespondanceEstimator.topologyExtractor.selfOrganizingMap.registerCallback(
            somVisualizationCallback
        )

        l1VisualizationCallback = setupVisualizationCallback(
            testCorrespondanceEstimator.topologyExtractor.l1Median
        )
        testCorrespondanceEstimator.topologyExtractor.l1Median.registerCallback(
            l1VisualizationCallback
        )
        print(
            "Corresponding branch indices: {}.".format(
                testCorrespondanceEstimator.getCorrespondingBranches()
            )
        )

    if vis:
        fig, ax = setupVisualization(
            testCorrespondanceEstimator.extractedTopology.X.shape[1]
        )
        pointPairs = (
            testCorrespondanceEstimator.extractedTopology.getAdjacentPointPairs()
        )
        leafNodeIndices = (
            testCorrespondanceEstimator.extractedTopology.getLeafNodeIndices()
        )
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
        plotPointSet(
            ax=ax,
            X=testCorrespondanceEstimator.extractedTopology.X,
            color=[1, 0, 0],
            size=30,
        )
        plotPointSet(
            ax=ax,
            X=testCorrespondanceEstimator.extractedTopology.X,
            color=[1, 0, 0],
            size=20,
        )
        plotPointSet(
            ax=ax,
            X=testCorrespondanceEstimator.extractedTopology.X[leafNodeIndices, :],
            color=[1, 0, 0],
            size=50,
            alpha=0.4,
        )
        set_axes_equal(ax)
        plt.show(block=True)


def test_correspondanceEstimation():
    dataPath = "tests/testdata/topologyExtraction/wireHarnessReduced.txt"
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
    templateTopology = testCorrespondanceEstimator.templateTopology
    extractedTopology = testCorrespondanceEstimator.extractedTopology
    branchMapping = testCorrespondanceEstimator.getCorrespondingBranches()

    # test corresponance calculation
    print(
        "Node correspondances: {}".format(
            extractedTopology.calculateCorrespondingNodesForPointSet(
                testCorrespondanceEstimator.Y
            )
        )
    )
    print(
        "Corresponding Brances: {}".format(
            extractedTopology.calculateCorrespondingBranchesForPointSet(
                testCorrespondanceEstimator.Y
            )
        )
    )
    print(
        "Branch correspondance of point set: {}".format(
            extractedTopology.getBranchCorrespondanceFromPointSet(
                testCorrespondanceEstimator.Y
            )
        )
    )
    testPointSet = testCorrespondanceEstimator.Y
    for i, branch in enumerate(extractedTopology.getBranches()):
        print(
            "Points corresponding to branch {}: {}".format(
                i, extractedTopology.getPointsCorrespondingToBranch(i, testPointSet)
            )
        )

    print(
        "InterpolatedCartesianPosition: {}".format(
            extractedTopology.interpolateCartesianPositionFromBranchLocalCoordinate(
                0, 0.1
            )
        )
    )
    (
        correspondingBranches,
        correspondingLocalCoordinates,
    ) = extractedTopology.calculateBranchCorrespondanceAndLocalCoordinatesForPointSet(
        testPointSet
    )
    print(
        "Corresponding branches: {}, corresponding local coordiantes {}".format(
            correspondingBranches, correspondingLocalCoordinates
        )
    )
    (
        correspondingBranches,
        correspondingLocalCoordinates,
    ) = testCorrespondanceEstimator.calculateTemplateBranchCorrespondanceAndLocalCoordinatsFromPointSet(
        testPointSet
    )
    print(
        "Corresponding branches: {}, corresponding local coordiantes {}".format(
            correspondingBranches, correspondingLocalCoordinates
        )
    )

    samplePoints = testCorrespondanceEstimator.X
    (
        samplePointsBranchIndices,
        samplePointsLocalCoordinates,
    ) = testCorrespondanceEstimator.calculateExtractedBranchCorrespondanceAndLocalCoordinatesFromPointSet(
        samplePoints
    )

    # test findCorrespondancesFromLocalCoordinate
    correspondingPointPairs = testCorrespondanceEstimator.getCorrespondingCartesianPointPairsFromBranchLocalCoordinatesInExtractedTopology(
        samplePointsBranchIndices, samplePointsLocalCoordinates
    )

    # (
    #     Ysample,
    #     Xsample,
    #     C,
    # ) = testCorrespondanceEstimator.findCorrespondancesFromLocalCoordinate(0.1)

    # test findCorrespondancesFromLocalCoordinates
    S = np.linspace(0, 1, 5)
    (
        Ysample,
        Xsample,
        C,
    ) = testCorrespondanceEstimator.findCorrespondancesFromLocalCoordinates(S)

    if vis:
        # colormap
        colorMap = matplotlib.colormaps["viridis"]
        lowerLim = 0
        upperLim = np.round(templateTopology.getNumBranches() - 1)
        norm = matplotlib.colors.Normalize(vmin=lowerLim, vmax=upperLim)  # Normalizer
        sm = plt.cm.ScalarMappable(cmap=colorMap, norm=norm)  # creating ScalarMappable

        templatePointPairs = (
            templateTopology.getAdjacentPointPairsAndBranchCorrespondance()
        )
        extractedPointPairs = (
            extractedTopology.getAdjacentPointPairsAndBranchCorrespondance()
        )
        # visualize template topology tree
        fig, ax = setupVisualization(len(templatePointPairs[0][0]))
        for pointPair in templatePointPairs:
            stackedPair = np.stack(pointPair[:2])
            branchNumber = pointPair[2]
            plotColor = [
                sm.to_rgba(branchNumber)[0],
                sm.to_rgba(branchNumber)[1],
                sm.to_rgba(branchNumber)[2],
            ]
            plotLine(ax, pointPair=stackedPair, color=plotColor)
            plotPoint(ax=ax, x=stackedPair[0], color=plotColor)
            plotPoint(ax=ax, x=stackedPair[1], color=plotColor)
        set_axes_equal(ax)
        plt.show(block=False)

        # visualize extracted topology tree
        fig, ax = setupVisualization(len(extractedPointPairs[0][0]))
        for pointPair in extractedPointPairs:
            stackedPair = np.stack(pointPair[:2])
            branchNumber = np.where(branchMapping == pointPair[2])[0][0]
            plotColor = [
                sm.to_rgba(branchNumber)[0],
                sm.to_rgba(branchNumber)[1],
                sm.to_rgba(branchNumber)[2],
            ]
            plotLine(ax, pointPair=stackedPair, color=plotColor)
            plotPoint(ax=ax, x=stackedPair[0], color=plotColor)
            plotPoint(ax=ax, x=stackedPair[1], color=plotColor)
        set_axes_equal(ax)
        plt.show(block=False)

        # visualize point correspondances
        fig, ax = setupVisualization(len(extractedPointPairs[0][0]))
        for pointPair in correspondingPointPairs:
            stackedPair = np.stack(pointPair[:2])
            plotLine(ax, pointPair=stackedPair, color=[1, 0, 0], alpha=0.3)
            plotPoint(ax=ax, x=stackedPair[0], color=[1, 0, 0])
            plotPoint(ax=ax, x=stackedPair[1], color=[0, 0, 1])
        set_axes_equal(ax)

        # visualize point correspondances for sampling methods
        fig, ax = setupVisualization(len(extractedPointPairs[0][0]))
        transformedX = C @ Xsample
        for i in range(len(Ysample)):
            stackedPair = np.stack((transformedX[i, :], Ysample[i, :]))
            plotLine(ax, pointPair=stackedPair, color=[1, 0, 0], alpha=0.3)
            plotPoint(ax=ax, x=stackedPair[0], color=[1, 0, 0])
            plotPoint(ax=ax, x=stackedPair[1], color=[0, 0, 1])
        set_axes_equal(ax)
        plt.show(block=True)


if __name__ == "__main__":
    # test_topologyExtraction()
    test_correspondanceEstimation()
