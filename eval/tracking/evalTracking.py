import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.evaluation.evaluation import Evaluation

    # tracking algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import KinematicsPreservingRegistration
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.tracking.krspr.krspr import (
        KinematicRegularizedStructurePreservedRegistration,
    )

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global vis
vis = True
save = False

# setup evalulation class
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = Evaluation(configFilePath=pathToConfigFile)
# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetsToLoad"][0]]
fileIdentifier = eval.config["filesToLoad"][0]
saveFolderPath = "data/eval/tracking/"

fileName = eval.getFileName(fileIdentifier, dataSetPath)
filePath = eval.getFilePath(fileIdentifier, dataSetPath)


def visualizationCallback(
    fig,
    ax,
    classHandle,
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
    eval.results[0]["result"]["T"].append(classHandle.T.copy())
    eval.results[0]["result"]["iteration"].append(classHandle.iteration)


def runEvaluation(dataSetPath):
    global eval
    # generate a model for the data set
    bdloModel = eval.generateModel(dataSetPath, 30)

    # setup result file
    result = {
        "dataSetPath": dataSetPath,
        "evalConfig": eval.config,
        "result": {
            "Y": [],
            "X": [],
            "T": [],
            "iteration": [],
            "runtimePerIteration": [],
        },
    }
    eval.results.append(result)

    # load first point cloud from data set
    pointCloud = eval.getPointCloud(
        0,
        dataSetPath,
    )
    Y = pointCloud[0]

    # setup registrations
    XInit = bdloModel.getCartesianBodyCenterPositions()
    qInit = bdloModel.getGeneralizedCoordinates()
    cpd = CoherentPointDrift(Y=Y, X=XInit, **eval.config["cpdParameters"])
    spr = StructurePreservedRegistration(Y=Y, X=XInit, **eval.config["sprParameters"])
    kpr = KinematicsPreservingRegistration(
        Y=Y,
        qInit=qInit,
        model=KinematicsModelDart(bdloModel.skel.clone()),
        **eval.config["kprParameters"],
    )
    if vis:
        visualizationCallback_cpd = eval.getVisualizationCallback(cpd)
        visualizationCallback_spr = eval.getVisualizationCallback(spr)
        visualizationCallback_kpr = eval.getVisualizationCallback(kpr)
        cpd.registerCallback(visualizationCallback_cpd)
        spr.registerCallback(visualizationCallback_spr)
        kpr.registerCallback(visualizationCallback_kpr)
    cpd.register()
    spr.register()
    kpr.register()
    for i in range(1, eval.getNumImageSetsInDataSet(dataSetPath)):
        pointCloud = eval.getPointCloud(
            i,
            dataSetPath,
        )
        Y = pointCloud[0]
        cpd.Y = Y
        spr.Y = Y
        kpr.Y = Y
        cpd.register()
        spr.register()
        kpr.register()
    return result


if __name__ == "__main__":
    # run evaluation
    runEvaluation(dataSetPath)

    # if save:
    #     # save evaluation data
    #     filePathToSaved = eval.saveResults(saveFolderPath)

    # # load last result for evaluation
    # if save:
    #     result = eval.loadResults(filePathToSaved)
    # else:
    #     result = eval.results[0]
    # # plot evaluation
    # somResult = result[0]["result"]
    # # tracking error
    # trackingErrors = []
    # Y = somResult["Y"][0]
    # for T in somResult["T"]:
    #     distanceMatrix = distance_matrix(T, Y)
    #     if Y.shape[0] >= T.shape[0]:
    #         correspodingIndices = np.argmin(distanceMatrix, axis=0)
    #         trackingError = 0
    #         for m in range(len(Y)):
    #             trackingError += distanceMatrix[correspodingIndices[m], m]
    #     else:
    #         correspodingIndices = np.argmin(distanceMatrix, axis=1)
    #         for n in range(len(T)):
    #             trackingError += np.sum(distanceMatrix[n, correspodingIndices[n]])
    #     trackingErrors.append(trackingError)
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.plot(list(range(len(trackingErrors))), trackingErrors)
    # plt.show(block=True)