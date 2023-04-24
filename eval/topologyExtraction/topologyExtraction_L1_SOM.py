import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/eval/topologyExtraction", ""))
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.visualization.plot3D import *
    from src.sensing.loadPointCloud import readPointCloudFromPLY
except:
    print("Imports for Evaluation of Topology Extraction Method L1_SOM failed.")
    raise

# control parameters
# source data
sourceSample = 1
dataSrc = [
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_7.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_4.ply",
]
dataPath = dataSrc[sourceSample]

vis = True  # enable for visualization
numSeedPointsL1 = 300
numSeedPointsSOM = 50


def setupTopologyExtractor(pointSet):
    somParameters = {
        "h0": 1,
        "h0Annealing": 1,
        "sigma2": 0.1,
        "sigma2Min": 0.01,
        "sigma2Annealing": 0.8,
        "method": "kernel",
        "max_iterations": 10,
    }

    l1Parameters = {
        "h": 0.01,
        "hAnnealing": 0.9,
        "hMin": 0.01,
        "hReductionFactor": 1,
        "mu": 0.35,
        "max_iterations": 30,
    }
    lofOutlierFilterParameters = {
        "numNeighbors": 15,
        "contamination": 0.1,
    }
    topologyExtractor = TopologyExtraction(
        **{
            "Y": pointSet,
            "somParameters": somParameters,
            "l1Parameters": l1Parameters,
            "lofOutlierFilterParameters": lofOutlierFilterParameters,
        }
    )
    if vis:
        l1VisualizationCallback = setupVisualizationCallback(topologyExtractor.l1Median)
        topologyExtractor.l1Median.registerCallback(l1VisualizationCallback)
        somVisualizationCallback = setupVisualizationCallback(
            topologyExtractor.selfOrganizingMap
        )
        topologyExtractor.selfOrganizingMap.registerCallback(somVisualizationCallback)
    return topologyExtractor


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


def evalTopologyExtraction():
    Y = readPointCloudFromPLY(dataPath)[:, :3]
    topologyExtractor = setupTopologyExtractor(Y)
    seedPointsL1 = topologyExtractor.randomSample(Y, numSeedPointsL1)
    reducedPointSetL1 = topologyExtractor.reducePointSetL1(Y, seedPointsL1)
    seedPointsSOM = topologyExtractor.randomSample(Y, numSeedPointsSOM)
    reducedPointSetSOM = topologyExtractor.reducePointSetSOM(
        reducedPointSetL1, seedPointsSOM
    )
    extractedTopology = topologyExtractor.extractTopology(
        reducedPointSetSOM, reducedPointSetL1, method="combined"
    )

    if vis:
        fig, ax = setupVisualization(topologyExtractor.X.shape[1])
        pointPairs = extractedTopology.getAdjacentPointPairs()
        leafNodeIndices = extractedTopology.getLeafNodeIndices()
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
        plotPointSet(ax=ax, X=extractedTopology.X, color=[1, 0, 0], size=30)
        plotPointSet(
            ax=ax,
            X=extractedTopology.X[leafNodeIndices, :],
            color=[1, 0, 0],
            size=50,
            alpha=0.4,
        )
        set_axes_equal(ax)
        plt.show(block=True)


if __name__ == "__main__":
    evalTopologyExtraction()
