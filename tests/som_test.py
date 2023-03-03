import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.dimreduction.som.som import SelfOrganizingMap
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.plot3D import plotPointSets
except:
    print("Imports for SOM Test failed.")
    raise

dataPath = "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_3D_DLO/pointcloud_1.ply"
# dataPath = "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply"
vis = True  # enable for visualization
# control parameters
alpha = 1
numNearestNeighbors = 5
numNearestNeighborsAnnealing = 0.7
sigma2 = 0.01
alphaAnnealing = 0.9
sigma2Annealing = 0.99
sampleRatio = 1 / 10
nthDataPoint = 2
numIterations = 10

fig = plt.figure()
ax = fig.add_subplot(projection="3d")


def setupVisualizationCallback(classHandle):

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
    print(classHandle.iteration)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def test_som():
    testCloud = readPointCloudFromPLY(dataPath)[::nthDataPoint, :3]
    numSeedpoints = int(len(testCloud) * sampleRatio)
    random_indices = random.sample(range(0, len(testCloud)), numSeedpoints)
    seedpoints = testCloud[random_indices, :]
    testReduction = SelfOrganizingMap(
        **{
            "Y": testCloud,
            "X": seedpoints,
            "alpha": alpha,
            "numNearestNeighbors": numNearestNeighbors,
            "numNearestNeighborsAnnealing": numNearestNeighborsAnnealing,
            "sigma2": sigma2,
            "alphaAnnealing": alphaAnnealing,
            "sigma2Annealing": sigma2Annealing,
            "max_iterations": numIterations,
        }
    )
    if vis:
        visCallback = setupVisualizationCallback(testReduction)
        testReduction.registerCallback(visCallback)
    reducedPoints = testReduction.calculateReducedRepresentation()
    print(reducedPoints)


if __name__ == "__main__":
    test_som()
