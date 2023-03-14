import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn import manifold
from scipy.spatial import distance_matrix
from sklearn import preprocessing

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.dimreduction.som.som import SelfOrganizingMap
    from src.dimreduction.l1median.l1Median import L1Median
    from src.dimreduction.mlle.mlle import Mlle
    from src.dimreduction.mlle.lle import Lle
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.localization.mst.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.plot3D import (
        plotPointSets,
        plotPointSet,
        plotPoint,
        plotLine,
        set_axes_equal,
    )
    from src.utils.utils import knn
    from src.utils.utils import sqdistance_matrix
except:
    print("Imports for application topologyReconstructionFromPointCloud failed.")
    raise

# script control parameters
# ------------------------------------------------------------------------
# visualization
visControl = {
    "visualizeInput": True,
    "visualizeRandomSample": True,
    "visualizeDimReducedPointSet": True,
    "visualizeReducedPointSet": True,
    "visualizeFilteredPointSet": True,
    "visualizeTopology": True,
}

# saving
save = False  # if data  should be saved under the given savepath
savePath = "tests/testdata/topologyExtraction/topologyExtractionTestSet.txt"

# source data
sourceSample = 1
dataSrc = [
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_3D_DLO/pointcloud_1.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_7.ply",
]
dataPath = dataSrc[sourceSample]

# downsampling
downsamplingInputRatio = 1 / 3  # downsampling of input point set
numSeedPoints = 70  # downsampling for obtaining seedpoints

# outlier filtering
numNeighbors = 15
contamination = 0.1

# downsampling algorithm parameters
somParameters = {
    "alpha": 1,
    "numNearestNeighbors": 30,
    "numNearestNeighborsAnnealing": 0.8,
    "sigma2": 0.03,
    "alphaAnnealing": 0.9,
    "sigma2Annealing": 0.8,
    "kernelMethod": False,
    "max_iterations": 3,
}

l1Parameters = {
    "h": 0.12,
    "hAnnealing": 0.8,
    "hReductionFactor": 1,
    "mu": 0.35,
    "max_iterations": 3,
}

# mlle parameters,
numSeedPoints_MLLE = numSeedPoints
mlleParameters = {
    "k": 30,
    "d": 2,
    "tol": 1e-3,
    "solverType": "dense",
    "mapping": "power",
    "sigma": 0.1,
    "exponent": 2,
}
# mlleParameters = {
#     "method": "modified",
#     "n_neighbors": int(0.9 * numSeedPoints),
#     "n_components": 2,
#     "eigen_solver": "auto",
#     "random_state": 0,
# }

# algorithm order
reductionOrder = {
    # "som": somParameters,
    "l1": l1Parameters
}


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
    plt.cla()
    plotPointSets(
        ax=ax,
        X=classHandle.T,
        Y=classHandle.Y,
        ySize=5,
        xSize=10,
        # yMarkerStyle=".",
        yAlpha=0.05,
    )
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)
    print(classHandle.iteration)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def readData(path):
    pointSet = readPointCloudFromPLY(path)[:: int((1 / downsamplingInputRatio)), :3]
    if visControl["visualizeInput"]:
        fig, ax = setupVisualization(pointSet.shape[1])
        plotPointSet(ax=ax, X=pointSet)
        set_axes_equal(ax)
        plt.show(block=False)
    return pointSet


def samplePointsRandom(pointSet, numSeedPoints):
    random_indices = random.sample(range(0, len(pointSet)), numSeedPoints)
    seedPoints = pointSet[random_indices, :]
    if visControl["visualizeRandomSample"]:
        fig, ax = setupVisualization(seedPoints.shape[1])
        plotPointSet(ax=ax, X=seedPoints)
        set_axes_equal(ax)
        plt.show(block=False)
    return seedPoints


def reduceDimension(pointSet, dimReductionPrameters: dict):
    dimReductionPrameters["X"] = pointSet
    mlle = Mlle(**dimReductionPrameters)
    reconstructedPointSet = mlle.solve()
    # mlle = Lle(**dimReductionPrameters)
    # reconstructedPointSet = mlle.solve()
    # mlle = manifold.LocallyLinearEmbedding(**mlleParameters)
    # reconstructedPointSet = mlle.fit_transform(pointSet)

    if (
        visControl["visualizeDimReducedPointSet"]
        and reconstructedPointSet.shape[1] == 3
    ):
        fig, ax = setupVisualization(reconstructedPointSet.shape[1])
        plotPointSet(ax=ax, X=reconstructedPointSet)
        set_axes_equal(ax)
        plt.show(block=False)
    elif (
        visControl["visualizeDimReducedPointSet"]
        and reconstructedPointSet.shape[1] == 2
    ):
        fig, ax = setupVisualization(reconstructedPointSet.shape[1])
        plotPointSet(ax=ax, X=reconstructedPointSet)
        set_axes_equal(ax)
        plt.show(block=False)
    elif (
        visControl["visualizeDimReducedPointSet"]
        and reconstructedPointSet.shape[1] == 1
    ):
        fig, ax = setupVisualization(reconstructedPointSet.shape[1])
        plotPointSet(ax=ax, X=np.hstack((reconstructedPointSet, reconstructedPointSet)))
        set_axes_equal(ax)
        plt.show(block=False)
    else:
        pass
    return (reconstructedPointSet, mlle)


def reducePointSet(
    pointSet, seedPoints, reductionMethod: str, reductionParameters: dict
):
    if reductionMethod == "som":
        reductionParameters["Y"] = pointSet
        reductionParameters["X"] = seedPoints
        myReduction = SelfOrganizingMap(**reductionParameters)
    elif reductionMethod == "l1":
        reductionParameters["Y"] = pointSet
        reductionParameters["X"] = seedPoints
        myReduction = L1Median(**reductionParameters)
    else:
        raise NotImplementedError

    if visControl["visualizeReducedPointSet"]:
        visCallback = setupVisualizationCallback(myReduction)
        myReduction.registerCallback(visCallback)

    reducedPoints = myReduction.calculateReducedRepresentation()
    return reducedPoints, myReduction


def filterOutliers(pointSet):
    lofFilter = LocalOutlierFactor(
        n_neighbors=numNeighbors, contamination=contamination
    )
    filterResult = lofFilter.fit_predict(pointSet)
    negOutlierScore = lofFilter.negative_outlier_factor_
    filteredPointSet = pointSet[np.where(filterResult != -1)[0], :]

    if visControl["visualizeFilteredPointSet"]:
        fig, ax = setupVisualization(pointSet.shape[1])
        for i, point in enumerate(pointSet):
            (negOutlierScore.max() - negOutlierScore[i]) / (
                negOutlierScore.max() - negOutlierScore.min()
            )
            if filterResult[i] == 1:
                color = np.array([0, 0, 1])
            else:
                color = np.array([1, 0, 0])
            plotPoint(ax=ax, x=point, size=2 * i, color=color, alpha=0.2)
        plt.show(block=False)

    return filteredPointSet


def extractTopology(pointSet, featureMatrix=None):
    topology = MinimalSpanningTreeTopology(
        **{
            "X": pointSet,
            "featureMatrix": featureMatrix,
        }
    )
    if visControl["visualizeTopology"]:
        fig, ax = setupVisualization(pointSet.shape[1])
        pointPairs = topology.getAdjacentPointPairs()
        leafNodeIndices = topology.getLeafNodeIndices()
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
        plotPointSet(ax=ax, X=pointSet, color=[1, 0, 0], size=30)
        plotPointSet(ax=ax, X=pointSet, color=[1, 0, 0], size=20)
        plotPointSet(
            ax=ax, X=pointSet[leafNodeIndices, :], color=[1, 0, 0], size=50, alpha=0.4
        )
        set_axes_equal(ax)
        plt.show(block=True)


def eval_SOM():
    inputPointSet = readData(dataPath)
    seedPoints = samplePointsRandom(inputPointSet, numSeedPoints)
    reducedPointSet = reducePointSet(inputPointSet, seedPoints, "som", somParameters)
    filteredPointSet = filterOutliers(reducedPointSet)
    extractTopology(filteredPointSet)


def eval_L1():
    inputPointSet = readData(dataPath)
    seedPoints = samplePointsRandom(inputPointSet, numSeedPoints)
    reducedPointSet = reducePointSet(inputPointSet, seedPoints, "l1", l1Parameters)
    filteredPointSet = filterOutliers(reducedPointSet)
    extractTopology(filteredPointSet)


def eval_MLLE():
    inputPointSet = readData(dataPath)
    samplePoints = samplePointsRandom(inputPointSet, numSeedPoints_MLLE)
    reconstrucedPointSet = reduceDimension(samplePoints, mlleParameters)
    plt.show(block=True)


def eval_MLLE_4D_2D():
    inputPointSet = readData(dataPath)
    samplePoints = samplePointsRandom(inputPointSet, numSeedPoints_MLLE)
    mlleParameters["d"] = 5
    reconstrucedPointSet = reduceDimension(samplePoints, mlleParameters)
    mlleParameters["d"] = 3
    reconstrucedPointSet = reduceDimension(reconstrucedPointSet, mlleParameters)
    plt.show(block=True)


def eval_SOM_L1():
    inputPointSet = readData(dataPath)
    seedPoints = samplePointsRandom(inputPointSet, numSeedPoints)
    reducedPointSet, _ = reducePointSet(inputPointSet, seedPoints, "som", somParameters)
    reducedPointSet, _ = reducePointSet(
        inputPointSet, reducedPointSet, "l1", l1Parameters
    )
    filteredPointSet = filterOutliers(reducedPointSet)
    extractTopology(filteredPointSet)


def eval_SOM_L1_MLLE():
    inputPointSet = readData(dataPath)
    seedPoints = samplePointsRandom(inputPointSet, numSeedPoints)
    reducedPointSet = reducePointSet(inputPointSet, seedPoints, "som", somParameters)
    reducedPointSet = reducePointSet(inputPointSet, reducedPointSet, "l1", l1Parameters)
    reconstructedPointSet = reduceDimension(reducedPointSet, mlleParameters)
    extractTopology(reconstructedPointSet)


def eval_MLLE_SOM_L1():
    inputPointSet = readData(dataPath)
    randomSample = samplePointsRandom(inputPointSet, numSeedPoints_MLLE)
    reconstructedPointSet = reduceDimension(randomSample, mlleParameters)
    seedPoints = samplePointsRandom(reconstructedPointSet, numSeedPoints)
    reducedPointSet = reducePointSet(
        reconstructedPointSet, seedPoints, "som", somParameters
    )
    reducedPointSet = reducePointSet(
        reconstructedPointSet, reducedPointSet, "l1", l1Parameters
    )
    filteredPointSet = filterOutliers(reducedPointSet)
    extractTopology(filteredPointSet)


def eval_LocalDensityBasedFeatureMatrix():
    inputPointSet = readData(dataPath)
    # dimreducedPointSet, mlle = reduceDimension(inputPointSet, mlleParameters)
    seedPoints = samplePointsRandom(inputPointSet, numSeedPoints)
    reducedPointSet, som = reducePointSet(
        inputPointSet, seedPoints, "som", somParameters
    )
    # filteredPointSet = filterOutliers(reducedPointSet)
    reducedPointSet, l1Median = reducePointSet(
        inputPointSet, reducedPointSet, "l1", l1Parameters
    )

    distances = distance_matrix(reducedPointSet, reducedPointSet)

    h_d = 0.001
    localDensities = np.ones((reducedPointSet.shape[0], reducedPointSet.shape[0]))
    for i, xi in enumerate(reducedPointSet):
        for j, xj in enumerate(reducedPointSet):
            densityCheckPoint = xi + 0.5 * (xj - xi)
            localDensity = 1 + np.sum(
                np.exp(
                    (
                        -(
                            distance_matrix(
                                np.reshape(
                                    densityCheckPoint, (-1, reducedPointSet.shape[1])
                                ),
                                inputPointSet,
                            )
                            ** 2
                        )
                    )
                    / ((h_d / 2) ** 2)
                )
            )
            localDensities[i, j] = localDensity

    # build feature matrix
    C = l1Median.getCorrespondences()

    featureMatrix = distances * localDensities

    extractTopology(reducedPointSet, featureMatrix)
    print("End")


def eval_SOM_L1_MLLEWeightedFeatureMatrix():
    inputPointSet = readData(dataPath)
    # dimreducedPointSet, mlle = reduceDimension(inputPointSet, mlleParameters)
    seedPoints = samplePointsRandom(inputPointSet, numSeedPoints)
    reducedPointSet, som = reducePointSet(
        inputPointSet, seedPoints, "som", somParameters
    )
    # filteredPointSet = filterOutliers(reducedPointSet)
    reducedPointSet, l1Median = reducePointSet(
        inputPointSet, reducedPointSet, "l1", l1Parameters
    )

    # build feature matrix
    # approach over Alignment Matrix
    # distanceMatrix = distance_matrix(reducedPointSet, reducedPointSet)
    # C = l1Median.getCorrespondences()
    # mlleDistances = np.zeros((reducedPointSet.shape[0], reducedPointSet.shape[0]))
    # Phi = mlle.Phi
    # for i, x1 in enumerate(reducedPointSet):
    #     for j, x2 in enumerate(reducedPointSet):
    #         mlleDistances[i, j] = np.sum(distance_matrix(Phi[C[i], :], Phi[C[j], :]))
    # featureMatrix = distanceMatrix * mlleDistances

    C = l1Median.getCorrespondences()

    # # find knn
    # (J, _) = knn(
    #     reducedPointSet, reducedPointSet, 12
    # )  # one more because knn includes each point itself.
    # mlleTransformParameters = {
    #     "k": 13 ,
    #     "d": 1,
    #     "tol": 1e-3,
    #     "solverType": "dense",
    # }
    # # approach over local MLLE
    # distanceMatrix = np.ones((reducedPointSet.shape[0], reducedPointSet.shape[0]))
    # mlleDistances = np.ones((reducedPointSet.shape[0], reducedPointSet.shape[0]))
    # for i, indices in enumerate(J):
    #     mlleIndices = {}
    #     knnCorrespondingPointList = [reducedPointSet[indices, :]]
    #     for j, idx in enumerate(indices):
    #         knnCorrespondingPointList.append(inputPointSet[C[idx], :])
    #         startIdx = len(mlleIndices)
    #         endIdx = startIdx + len(C[idx])
    #         mlleIndices[str(idx)] = list(range(startIdx, endIdx))

    #     (mlleTransformedCorrespondingPoints, _) = reduceDimension(
    #         np.vstack(knnCorrespondingPointList), mlleTransformParameters
    #     )
    #     sumMlleDistances = 0
    #     sumSpatialDistances = 0
    #     for j, idx in enumerate(indices):
    #         mlleDistances[i, idx] = np.linalg.norm(
    #             mlleTransformedCorrespondingPoints[0, :]
    #             - mlleTransformedCorrespondingPoints[j, :]
    #         )
    #         sumMlleDistances += mlleDistances[i, idx]
    #         distanceMatrix[i, idx] = np.linalg.norm(
    #             reducedPointSet[i, :] - reducedPointSet[idx, :]
    #         )
    #         sumSpatialDistances += distanceMatrix[i, idx]

    #     for j, idx in enumerate(indices):
    #         mlleDistances[i, idx] = mlleDistances[i, idx] / sumMlleDistances
    #         distanceMatrix[i, idx] = distanceMatrix[i, idx] / sumSpatialDistances

    # mlleCenter_i = (
    #     1
    #     / len(mlleIndices[str(i)])
    #     * np.sum(mlleTransformedCorrespondingPoints[mlleIndices[str(i)]])
    # )
    # sumMlleDistances = 0
    # for j, idx in enumerate(indices):
    #     mlleCenter_idx = (
    #         1
    #         / len(mlleIndices[str(idx)])
    #         * np.sum(mlleTransformedCorrespondingPoints[mlleIndices[str(idx)]])
    #     )
    #     mlleDistances[i, idx] = np.linalg.norm(mlleCenter_idx - mlleCenter_i)
    #     sumMlleDistances += mlleDistances[i, idx]

    # for j, idx in enumerate(indices):
    #     mlleDistances[i, idx] = mlleDistances[i, idx] / sumMlleDistances

    # approach: adding alignment matrix as a feature
    # create combined point set
    mlleTestParams = {
        "k": 300,
        "d": 3,
        "tol": 1e-3,
        "solverType": "dense",
    }
    combinedPointSet = np.vstack((reducedPointSet, inputPointSet))
    dimreducedPointSet, mlle = reduceDimension(combinedPointSet, mlleTestParams)
    min_max_scaler = preprocessing.MinMaxScaler()
    reducedPointSetNormalized = min_max_scaler.fit_transform(reducedPointSet)
    dimreducedPointSetNormalized = min_max_scaler.fit_transform(dimreducedPointSet)
    mlleAlignmentMatrixNormalized = preprocessing.normalize(
        mlle.Phi[0 : len(reducedPointSet), :]
    )
    mlleAlignmentFeature = (
        np.ones(mlleAlignmentMatrixNormalized.shape) - mlleAlignmentMatrixNormalized
    )
    # featureVector = np.hstack(
    #     (
    #         reducedPointSetNormalized,
    #         dimreducedPointSetNormalized[0 : len(reducedPointSet), :],
    #     )
    # )
    featureVector = np.hstack(
        (
            reducedPointSetNormalized,
            mlleAlignmentFeature,
        )
    )
    featureMatrix = distance_matrix(featureVector, featureVector)

    extractTopology(reducedPointSet, featureMatrix)
    print("End")


if __name__ == "__main__":
    # eval_SOM()
    # eval_L1()
    # eval_MLLE()
    eval_SOM_L1()
    # eval_SOM_L1_MLLE()  # seems not useful
    # eval_MLLE_SOM_L1()
    # eval_MLLE_4D_2D()
    # eval_SOM_L1_MLLEWeightedFeatureMatrix()
    # eval_LocalDensityBasedFeatureMatrix()
