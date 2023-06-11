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
    from src.tracking.krcpd.krcpd import (
        KinematicRegularizedCoherentPointDrift,
    )

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global vis
global result
vis = True
save = True
loadInitialStateFromResult = True

# setup evalulation class
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = Evaluation(configFilePath=pathToConfigFile)
# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetsToLoad"][0]]
dataSetName = eval.config["dataSetPaths"][0].split("/")[-2]
fileIdentifier = eval.config["filesToLoad"][0]
resultFolderPath = "data/eval/tracking/" + dataSetName + "/"
resultFileName = "result"
resultFilePath = resultFolderPath + resultFileName
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


# def initialLocalization(bdloModel, pointSet):
# align the inital pose of the model
# extract inital pose
# localCoordinateSamples = np.linspace(
#     0,
#     1,
#     localizationParameters["numLocalCoordinateSamples"],
# )
# localization = BDLOLocalization(
#     **{
#         "Y": Y,
#         "S": localCoordinateSamples,
#         "templateTopology": bdloModel,
#         "extractedTopology": extractedTopology,
#     }
# qInit = localization.reconstructShape(
#     numIter=localizationParameters["numIter"], verbose=2
# )
# bdloModel.align(Y)


def setupResultTemplate(dataSetPath):
    # setup result file
    resultTemplate = {
        "dataSetPath": dataSetPath,
        "evalConfig": eval.config,
        "modelGeneration": {},
        "initialization": {
            "pointCloud": None,
            "topologyExtraction": {},
            "localization": {},
        },
        "tracking": {},
    }
    return resultTemplate


def setupTrackingResultTemplate():
    trackingResultTemplate = {
        "method": "",
        "results": [],
    }


def setupRegistrationResultTemplate():
    # setup tracking result
    registrationResultTemplate = {
        "Y": None,
        "X": None,
        "T": [],
        "iteration": [],
        "runtimePerIteration": [],
    }
    return registrationResultTemplate


def runModelGeneration(dataSetPath):
    bdloModel = eval.generateModel(
        dataSetPath, eval.config["modelGeneration"]["numSegments"]
    )
    return bdloModel


def loadPointCloud(dataSetPath, fileNumber):
    pointCloud = eval.getPointCloud(
        fileNumber,
        dataSetPath,
    )
    return pointCloud


def runTopologyExtraction(pointSet):
    # extract topology
    extractedTopology, topologyExtraction = eval.extractTopology(
        pointSet,
        somParameters=eval.config["topologyExtraction"]["somParameters"],
        l1Parameters=eval.config["topologyExtraction"]["l1Parameters"],
        pruningThreshold=eval.config["topologyExtraction"]["pruningThreshold"],
        skeletonize=True,
        visualizeSOMIteration=True,
        visualizeSOMResult=True,
        visualizeL1Iterations=True,
        visualizeL1Result=True,
        visualizeExtractionResult=True,
    )
    # eval.results[0]["initialization"]["topologyExtraction"][
    #     "topologyExtraction"
    # ] = topologyExtraction
    eval.results[0]["initialization"]["topologyExtraction"][
        "extractedTopology"
    ] = extractedTopology
    somResult = {
        "X": topologyExtraction.selfOrganizingMap.X,
        "Y": topologyExtraction.selfOrganizingMap.Y,
        "T": topologyExtraction.selfOrganizingMap.T,
    }
    l1Result = {
        "X": topologyExtraction.l1Median.X,
        "Y": topologyExtraction.l1Median.Y,
        "T": topologyExtraction.l1Median.T,
    }
    eval.results[0]["initialization"]["topologyExtraction"]["som"] = somResult
    eval.results[0]["initialization"]["topologyExtraction"]["l1"] = l1Result
    return extractedTopology


def runInitialLocalization(dataSetPath):
    # load first point cloud of the data set
    pointCloud = loadPointCloud(dataSetPath, 0)
    eval.results[0]["initialization"]["pointCloud"] = pointCloud
    Y = pointCloud[0]
    extractedTopology = runTopologyExtraction(Y)
    # get the model
    # bdloModel = eval.results[0]["modelGeneration"]["model"]
    bdloModel = runModelGeneration(dataSetPath)
    # perform initial localization
    XResult, qResult, localization = eval.initialLocalization(
        pointSet=Y,
        extractedTopology=extractedTopology,
        bdloModel=bdloModel,
        numSamples=eval.config["localization"]["numSamples"],
        numIterations=eval.config["localization"]["numIterations"],
        verbose=eval.config["localization"]["verbose"],
        method=eval.config["localization"]["method"],
        jacobianDamping=eval.config["localization"]["jacobianDamping"],
        visualizeCorresponanceEstimation=True,
        visualizeIterations=True,
        visualizeResult=True,
        visualizationCallback=None,
        block=False,
    )
    # eval.results[0]["initialization"]["localization"]["localization"] = localization
    eval.results[0]["initialization"]["localization"]["S"] = localization.S
    eval.results[0]["initialization"]["localization"]["C"] = localization.C
    eval.results[0]["initialization"]["localization"]["XLog"] = localization.XLog
    eval.results[0]["initialization"]["localization"]["qLog"] = localization.qLog
    eval.results[0]["initialization"]["localization"]["XResult"] = XResult
    eval.results[0]["initialization"]["localization"]["qResult"] = qResult
    return


def runEvaluation(dataSetPath):
    # get the model
    # bdloModel = eval.results[0]["modelGeneration"]["model"]
    bdloModel = runModelGeneration(dataSetPath)
    # setup registrations
    XInit = eval.results[0]["initialization"]["localization"]["XResult"]
    qInit = eval.results[0]["initialization"]["localization"]["qResult"]
    Y = eval.results[0]["initialization"]["pointCloud"][0]
    # # cpd
    # cpd = CoherentPointDrift(Y=Y, X=XInit, **eval.config["cpdParameters"])
    # if vis:
    #     visualizationCallback_cpd = eval.getVisualizationCallback(cpd)
    #     cpd.registerCallback(visualizationCallback_cpd)
    # cpd.registerCallback(visualizationCallback_cpd)
    # for i in range(1, eval.getNumImageSetsInDataSet(dataSetPath)):
    #     pointCloud = eval.getPointCloud(
    #         i,
    #         dataSetPath,
    #     )
    #     Y = pointCloud[0]
    #     cpd.Y = Y
    #     cpd.register(checkConvergence=False)

    # # spr
    # spr = StructurePreservedRegistration(Y=Y, X=XInit, **eval.config["sprParameters"])
    # if vis:
    #     visualizationCallback_spr = eval.getVisualizationCallback(spr)
    #     spr.registerCallback(visualizationCallback_spr)
    # spr.register(checkConvergence=False)

    # for i in range(1, eval.getNumImageSetsInDataSet(dataSetPath)):
    #     pointCloud = eval.getPointCloud(
    #         i,
    #         dataSetPath,
    #     )
    #     Y = pointCloud[0]
    #     spr.Y = Y
    #     spr.register(checkConvergence=False)

    # # kpr
    # kpr = KinematicsPreservingRegistration(
    #     Y=Y,
    #     qInit=qInit,
    #     model=KinematicsModelDart(bdloModel.skel.clone()),
    #     **eval.config["kprParameters"],
    # )
    # if vis:
    #     visualizationCallback_kpr = eval.getVisualizationCallback(kpr)
    #     kpr.registerCallback(visualizationCallback_kpr)
    # kpr.register(checkConvergence=False)
    # for i in range(1, eval.getNumImageSetsInDataSet(dataSetPath)):
    #     pointCloud = eval.getPointCloud(
    #         i,
    #         dataSetPath,
    #     )
    #     Y = pointCloud[0]
    #     kpr.Y = Y
    #     kpr.register(checkConvergence=False)

    krcpd = KinematicRegularizedCoherentPointDrift(
        Y=Y,
        qInit=qInit,
        model=KinematicsModelDart(bdloModel.skel.clone()),
        **eval.config["krcpdParameters"],
    )
    if vis:
        visualizationCallback_krcpd = eval.getVisualizationCallback(krcpd)
        krcpd.registerCallback(visualizationCallback_krcpd)
    krcpd.register(checkConvergence=False)
    for i in range(1, eval.getNumImageSetsInDataSet(dataSetPath)):
        pointCloud = eval.getPointCloud(
            i,
            dataSetPath,
        )
        Y = pointCloud[0]
        krcpd.Y = Y
        krcpd.register(checkConvergence=False)
    return


if __name__ == "__main__":
    if not loadInitialStateFromResult:
        # setup result file
        result = setupResultTemplate(dataSetPath)
        eval.results.append(result)
        runModelGeneration(dataSetPath)
        runInitialLocalization(dataSetPath)
    else:
        results = eval.loadResults(resultFilePath + ".pkl")
        eval.results = results
    # run tracking evaluation
    trackingResult = runEvaluation(dataSetPath)
    # save results
    if save:
        filePathToSaved = eval.saveResults(
            folderPath=resultFolderPath, generateUniqueID=False, fileName=resultFileName
        )

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
