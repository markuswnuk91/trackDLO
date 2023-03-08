import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.dimreduction.l1median.l1Median import L1Median
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.plot3D import plotPointSets
except:
    print("Imports for L1Median Test failed.")
    raise

dataPath = "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_3D_DLO/pointcloud_1.ply"
dataPath = "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply"
vis = True  # enable for visualization
# control parameters
h = 0.1
mu = 0.3
sampleRatio = 1 / 100
numIterations = 10


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


def test_l1Median():
    testCloud = readPointCloudFromPLY(dataPath)[:, :3]
    numSeedpoints = int(len(testCloud) * sampleRatio)
    random_indices = random.sample(range(0, len(testCloud)), numSeedpoints)
    seedpoints = testCloud[random_indices, :]
    testReduction = L1Median(
        **{
            "Y": testCloud,
            "X": seedpoints,
            "h": h,
            "mu": mu,
            "max_iterations": numIterations,
        }
    )
    if vis:
        visCallback = setupVisualizationCallback(testReduction)
        testReduction.registerCallback(visCallback)
    l1MedianPoints = testReduction.calculateReducedRepresentation()
    print(l1MedianPoints)


if __name__ == "__main__":
    test_l1Median()
