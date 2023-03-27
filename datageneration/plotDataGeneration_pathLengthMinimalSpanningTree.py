import os, sys
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.visualization.plot3D import *
    from src.sensing.loadPointCloud import readPointCloudFromPLY
except:
    print("Imports for plot data generation: Path Length Minimal Spanning Tree failed.")
    raise

# script control parameters
save = False  # if data  should be saved under the given savepath
vis = True  # enable for visualization
savePath_Xskel = "plot/plotdata/pathLengthMinimalSpanningTree/Xskel.txt"
savePath_Y = "plot/plotdata/pathLengthMinimalSpanningTree/Y.txt"

# source data
sourceSample = 1
dataSrc = [
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_3D_DLO/pointcloud_1.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_7.ply",
    "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_4.ply",
]
dataPath = dataSrc[sourceSample]


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


def setupTopologyExtractor(pointSet):
    somParameters = {
        "alpha": 1,
        "numNearestNeighbors": 5,
        "numNearestNeighborsAnnealing": 0.95,
        "sigma2": 0.03,
        "alphaAnnealing": 0.9,
        "sigma2Annealing": 0.8,
        "kernelMethod": False,
        "max_iterations": 30,
    }

    l1Parameters = {
        "h": 0.12,
        "hAnnealing": 1,
        "hReductionFactor": 0.8,
        "mu": 0.35,
        "max_iterations": 300,
    }
    lofOutlierFilterParameters = {
        "numNeighbors": 15,
        "contamination": 0.1,
    }
    topologyExtractor = TopologyExtraction(
        **{
            "Y": pointSet,
            "numSeedPoints": 100,
            "somParameters": somParameters,
            "l1Parameters": l1Parameters,
            "lofOutlierFilterParameters": lofOutlierFilterParameters,
        }
    )
    return topologyExtractor


def generateData():
    Y = readPointCloudFromPLY(dataPath)[:, :3]
    topologyExtractor = setupTopologyExtractor(Y)
    if vis:
        somVisualizationCallback = setupVisualizationCallback(
            topologyExtractor.selfOrganizingMap
        )
        topologyExtractor.selfOrganizingMap.registerCallback(somVisualizationCallback)

        l1VisualizationCallback = setupVisualizationCallback(topologyExtractor.l1Median)
        topologyExtractor.l1Median.registerCallback(l1VisualizationCallback)
        minSpanTree_Xskel = topologyExtractor.extractTopology()
    if vis:
        fig, ax = setupVisualization(topologyExtractor.T.shape[1])
        plotPointSet(
            ax=ax,
            X=topologyExtractor.T,
            color=[0, 0, 0],
            size=30,
        )
        plotPointSet(
            ax=ax,
            X=topologyExtractor.Y,
            color=[0, 0, 0],
            size=5,
        )
        set_axes_equal(ax)
        plt.show(block=True)

    userInput = input("Ener y to save data: ")
    if userInput == "y":
        np.savetxt(savePath_Xskel, topologyExtractor.T)
        np.savetxt(savePath_Y, topologyExtractor.Y)
        print("Saved!")


if __name__ == "__main__":
    generateData()
