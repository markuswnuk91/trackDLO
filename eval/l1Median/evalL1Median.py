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
    from src.localization.downsampling.l1median.l1Median import L1Median
    from src.evaluation.evaluation import Evaluation

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global eval
global vis
vis = True
save = False
saveFolderPath = "data/eval/SOM/"


def visualizationCallback(
    fig,
    ax,
    classHandle,
    pauseInterval,
    savePath=None,
    fileName="img",
):
    global eval
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
        xColor=[0, 0, 1],
        yColor=[1, 0, 0],
    )
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)
    eval.results["result"]["T"].append(classHandle.T.copy())
    eval.results["result"]["iteration"].append(classHandle.iteration)


def runEvaluation(points, parameters):
    global eval
    l1Median = L1Median(**parameters)
    if parameters["numSeedPoints"] == -1:
        seedPoints = points
    else:
        seedPoints = l1Median.sampleRandom(points)
    # add inital state to result file
    eval.results["result"]["X"].append(seedPoints)
    eval.results["result"]["Y"].append(points)
    if vis:
        callback = eval.getVisualizationCallback(l1Median, visualizationCallback)
        l1Median.registerCallback(callback)
    resultingPoints = l1Median.calculateReducedRepresentation(points, seedPoints)
    return resultingPoints


if __name__ == "__main__":
    # read config and setup evaluation
    pathToConfigFile = (
        os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
    )
    eval = Evaluation(configFilePath=pathToConfigFile)

    # load point cloud data
    dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetsToLoad"][0]]
    fileIdentifier = eval.config["filesToLoad"][0]
    fileName = eval.getFileName(fileIdentifier, dataSetPath)
    filePath = eval.getFilePath(fileIdentifier, dataSetPath)
    dataSetFilePath = dataSetPath + fileName

    pointCloud = eval.getPointCloud(
        fileName,
        dataSetPath,
    )
    # setup result file
    result = {
        "dataSetFilePath": filePath,
        "evalConfig": eval.config,
        "result": {
            "Y": [],
            "X": [],
            "T": [],
            "iteration": [],
            "runtimePerIteration": [],
        },
    }
    eval.results = result

    # run evaluation
    runEvaluation(pointCloud[0], eval.config["l1Parameters"])

    if save:
        # save evaluation data
        filePathToSaved = eval.saveResults(saveFolderPath)

    # load last result for evaluation
    if save:
        result = eval.loadResults(filePathToSaved)
    else:
        result = eval.results
    # plot evaluation
    l1Result = result["result"]
    # tracking error
    trackingErrors = []
    Y = l1Result["Y"][0]
    for T in l1Result["T"]:
        distanceMatrix = distance_matrix(T, Y)
        if Y.shape[0] >= T.shape[0]:
            correspodingIndices = np.argmin(distanceMatrix, axis=0)
            trackingError = 0
            for m in range(len(Y)):
                trackingError += distanceMatrix[correspodingIndices[m], m]
        else:
            correspodingIndices = np.argmin(distanceMatrix, axis=1)
            for n in range(len(T)):
                trackingError += np.sum(distanceMatrix[n, correspodingIndices[n]])
        trackingErrors.append(trackingError)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(list(range(len(trackingErrors))), trackingErrors)
    plt.show(block=True)
