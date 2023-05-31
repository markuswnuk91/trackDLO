import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dartpy as dart
from scipy.spatial import distance_matrix
from functools import partial

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.evaluation.evaluation import Evaluation

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

# results
results = {
    "preprocessing": [],
    "topologyExtraction": [],
    "localization": [],
    "tracking": [],
}
global eval
global vis  # visualization
vis = True


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
        ySize=1,
        xSize=30,
        xColor=[1, 0, 0],
        yColor=[0, 0, 0],
    )
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)


def runEvaluation(points, parameters):
    global eval
    som = SelfOrganizingMap(**parameters)
    seedPoints = som.sampleRandom(points)
    if vis:
        callback = eval.getVisualizationCallback(som, visualizationCallback)
        som.registerCallback(callback)
    resultingPoints = som.calculateReducedRepresentation(points, seedPoints)
    return resultingPoints


if __name__ == "__main__":
    # read config and setup evaluation
    pathToConfigFile = (
        os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
    )
    eval = Evaluation(configFilePath=pathToConfigFile)
    # load point cloud data
    pointCloud = eval.getPointCloud(
        eval.config["filesToLoad"][0],
        eval.config["dataSetPaths"][eval.config["dataSetsToLoad"][0]],
    )
    # run evaluation
    runEvaluation(pointCloud[0], eval.config["somParameters"])
