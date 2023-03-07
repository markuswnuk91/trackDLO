import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.dimreduction.som.som import SelfOrganizingMap
    from src.dimreduction.l1median.l1Median import L1Median
    from src.dimreduction.mlle.mlle import Mlle
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.plot3D import plotPointSets, plotPointSet
except:
    print("Imports for application topologyReconstructionFromPointCloud failed.")
    raise

# script control parameters
# ------------------------------------------------------------------------
# visualization
visControl = {
    "visualizeInput": True,
    "visualizeReducedPointSet": True,
    "visualizeFilteredPointSet": True,
    "visualizeTopology": True,
}

# saving
save = False  # if data  should be saved under the given savepath
savePath = "tests/testdata/topologyExtraction/topologyExtractionTestSet.txt"

# source data
dataSrc = [
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_3D_DLO/pointcloud_1.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply",
]
dataPath = dataSrc[0]

# downsampling
downsamplingInputRatio = 1 / 3  # downsampling of input point set
downsamplingSeedPointRatio = 1 / 3  # downsampling for obtaining seedpoints

# outlier filtering
numNeighbors = 10
contamination = 0.1

# algorithm parameters
reductionMethod = "som"  # som, l1
somParameters = {
    "alpha": 1,
    "numNearestNeighbors": 10,
    "numNearestNeighborsAnnealing": 0.7,
    "sigma2": 0.01,
    "alphaAnnealing": 0.9,
    "sigma2Annealing": 0.99,
    "kernelMethod": True,
    "max_iterations": 3,
}


def setupVisualization():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # set axis limits
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.6)

    return fig, ax


def setupVisualizationCallback(classHandle):
    fig, ax = setupVisualization()
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
        waitTime=0.1,
    )
    print(classHandle.iteration)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def readData(path):
    pointSet = readPointCloudFromPLY(path)[:: int((1 / downsamplingInputRatio)), :3]
    if visControl["visualizeInput"]:
        fig, ax = setupVisualization()
        plotPointSet(ax=ax, X=pointSet, waitTime=-1)
    return pointSet


def reducePointSet(pointSet):
    numSeedpoints = int(len(pointSet) * downsamplingSeedPointRatio)
    random_indices = random.sample(range(0, len(pointSet)), numSeedpoints)
    seedpoints = pointSet[random_indices, :]
    if reductionMethod == "som":
        somParameters["Y"] = pointSet
        somParameters["X"] = seedpoints
        myReduction = SelfOrganizingMap(**somParameters)
    else:
        raise NotImplementedError

    if visControl["visualizeReducedPointSet"]:
        visCallback = setupVisualizationCallback(myReduction)
        myReduction.registerCallback(visCallback)

    reducedPoints = myReduction.calculateReducedRepresentation()
    return reducedPoints


def filterOutliers(pointSet):
    lofFilter = LocalOutlierFactor(
        n_neighbors=numNeighbors, contamination=contamination
    )
    filterResult = lofFilter.fit_predict(pointSet)
    negOutlierScore = lofFilter.negative_outlier_factor_
    filteredPointSet = pointSet[np.where(filterResult != -1)[0], :]

    if visControl["visualizeFilteredPointSet"]:
        fig, ax = setupVisualization()
        for i, point in enumerate(pointSet):
            (negOutlierScore.max() - negOutlierScore[i]) / (
                negOutlierScore.max() - negOutlierScore.min()
            )
            if filterResult[i] == 1:
                color = np.array([0, 0, 1])
            else:
                color = np.array([1, 0, 0])
            ax.scatter(point[0], point[1], point[2], s=2 * i, color=color, alpha=0.2)
        plt.show(block=False)

    return filteredPointSet


def extractTopology(pointSet):
    topology = TopologyExtraction(
        **{
            "X": pointSet,
        }
    )
    if visControl["visualizeTopology"]:
        fig, ax = setupVisualization()
        pointPairs = topology.getAdjacentPointPairs()
        leafNodeIndices = topology.getLeafNodeIndices()
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            ax.plot3D(stackedPair[:, 0], stackedPair[:, 1], stackedPair[:, 2], "blue")
        ax.scatter(
            pointSet[:, 0],
            pointSet[:, 1],
            pointSet[:, 2],
        )
        ax.scatter(pointSet[0, 0], pointSet[0, 1], pointSet[0, 2], "red", s=30)
        for i, leafPointIdx in enumerate(leafNodeIndices):
            ax.scatter(
                pointSet[leafPointIdx, 0],
                pointSet[leafPointIdx, 1],
                pointSet[leafPointIdx, 2],
                "yellow",
                s=(i + 1) * 300,
                alpha=0.4,
            )
        plt.show(block=True)


if __name__ == "__main__":
    inputPointSet = readData(dataPath)
    reducedPointSet = reducePointSet(inputPointSet)
    filteredPointSet = filterOutliers(reducedPointSet)
    extractTopology(filteredPointSet)
