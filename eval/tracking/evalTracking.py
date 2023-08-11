import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for tracking evaluation script failed.")
    raise

global runOpt
global visOpt
global saveOpt
runOpt = {"localization": False, "tracking": True, "evaluation": True}
visOpt = {
    "som": False,
    "somIterations": True,
    "l1": False,
    "l1Iterations": True,
    "topologyExtraction": True,
    "inverseKinematicsIterations": True,
    "correspondanceEstimation": False,
    "initializationResult": False,
    "trackingIterations": True,
}
saveOpt = {
    "localizationResults": False,
    "trackingResults": False,
    "evaluationResults": False,
}
loadInitialStateFromResult = True
runExperiment = True
registrationsToRun = [
    "cpd",
    "spr",
    "krcpd",
    "krcpd4BDLO",
]  # cpd, spr, krcpd, krcpd4BDLO
dataSetsToLoad = [0]  # -1 to load all data sets

savePath = "data/eval/tracking/results/"
resultFileName = "result"
dataSetPaths = [
    "data/darus_data_download/data/20230517_093927_manipulationsequence_manual_labeled_yshape/",
]


# def evaluateResults(result):
#     # compute tracking errors
#     for key in result["tracking"]:
#         trackingResults = result["tracking"][key]["results"]
#         trackingErrors = []
#         for trackingResult in trackingResults:
#             T = trackingResult["T"][-1]
#             Y = trackingResult["Y"]
#             trackingError = np.sum(distance_matrix(T, Y))
#             trackingErrors.append(trackingError)
#         # add to error to result file
#         result["tracking"][key]["trackingError"] = trackingErrors.copy()

#     # plot tracking errors
#     fig = plt.figure()
#     ax = plt.axes()
#     for key in result["tracking"]:
#         trackingErrors = result["tracking"][key]["trackingError"]
#         ax.plot(list(range(len(trackingErrors))), trackingErrors)
#     plt.show(block=True)


def calculateTrackingErrors(trackingResult):
    frames = []
    trackingErrors = []
    for registrationResult in trackingResult["registrations"]:
        frames.append(
            eval.getFileIdentifierFromFilePath(registrationResult["filePath"])
        )
        T = registrationResult["T"]
        Y = registrationResult["Y"]
        trackingError = 1 / len(Y) * np.sum(np.min(distance_matrix(T, Y), axis=0))
        trackingErrors.append(trackingError)
    return trackingErrors


def calculateGeometricErrors(trackingResult):
    accumulatedGeometricErrorPerIteration = []
    meanGeometricErrorPerIteration = []
    model = eval.generateModel(trackingResult["modelParameters"])
    B = trackingResult["B"]
    branchIndices = list(set(B))
    numBranches = len(branchIndices)
    totalLength = 0
    for branch in model.getBranches():
        totalLength += branch.getBranchInfo()["length"]
    correspondingNodeIndices = []
    XRef = model.getCartesianBodyCenterPositions()
    for branchIndex in branchIndices:
        nodeIndices = [i for i, x in enumerate(B) if x == branchIndex]
        correspondingNodeIndices.append(nodeIndices)

    registrationResults = trackingResult["registrations"]
    XInit = registrationResults[0]["X"]
    for registrationResult in registrationResults:
        T = registrationResult["T"]
        geometricBranchErrors = []
        desiredNodeDistances = []
        registeredNodeDistances = []
        referenceNodeDistances = []
        for i, branchIndex in enumerate(branchIndices):
            correspondingNodes = correspondingNodeIndices[i]
            correspondingT = T[correspondingNodes, :]
            correspondingXInit = XInit[correspondingNodes, :]
            correspondingXRef = XRef[correspondingNodes, :]
            referenceDifferences = np.diff(correspondingXRef, axis=0)
            currentDifferences = np.diff(correspondingT, axis=0)
            desiredDifferences = np.diff(correspondingXInit, axis=0)
            currentDistances = np.linalg.norm(currentDifferences, axis=1)
            desiredDistances = np.linalg.norm(desiredDifferences, axis=1)
            referenceDistances = np.linalg.norm(referenceDifferences, axis=1)
            for desiredNodeDistance, currentNodeDistance, referenceNodeDistance in zip(
                desiredDistances, currentDistances, referenceDistances
            ):
                registeredNodeDistances.append(currentNodeDistance)
                desiredNodeDistances.append(desiredNodeDistance)
                referenceNodeDistances.append(referenceNodeDistance)
            currentBranchLength = np.sum(currentDistances)
            desiredBranchLength = np.sum(desiredDistances)
            geometricBranchError = np.abs(desiredBranchLength - currentBranchLength)
            geometricBranchErrors.append(geometricBranchError)
        geometricError = np.sum(
            np.abs(np.array(desiredNodeDistances) - np.array(registeredNodeDistances))
        )
        meanGeometricError = np.mean(
            np.abs(np.array(desiredNodeDistances) - np.array(registeredNodeDistances))
        )
        accumulatedGeometricErrorPerIteration.append(geometricError)
        meanGeometricErrorPerIteration.append(meanGeometricError)
    return accumulatedGeometricErrorPerIteration, meanGeometricErrorPerIteration


def evaluateTrackingResults(results):
    trackingEvaluationResults = {}
    for trackingMethodResult in results["trackingResults"]:
        trackingEvaluationResult = {}
        method = trackingMethodResult["method"]
        trackingEvaluationResult["trackingResult"] = trackingMethodResult
        # tracking errors
        trackingErrors = calculateTrackingErrors(trackingMethodResult)
        trackingEvaluationResult["trackingErrors"] = trackingErrors
        # geometric errors
        (
            accumulatedGeometricErrorPerIteration,
            meanGeometricErrorPerIteration,
        ) = calculateGeometricErrors(trackingMethodResult)
        trackingEvaluationResult[
            "accumulatedGeometricErrors"
        ] = accumulatedGeometricErrorPerIteration
        trackingEvaluationResult["meanGeometricErrors"] = meanGeometricErrorPerIteration

        trackingEvaluationResults[method] = trackingEvaluationResult

        # reprojection errors
        print(
            "here we evaluate certain samples of frames in the data set for their reprojection error"
        )

    # successfully tracked frames
    print("here the images are generated to determine the frame until tracking fails")

    # runtime
    print("runtime is evaluated here")
    return trackingEvaluationResults


if __name__ == "__main__":
    if dataSetsToLoad[0] == -1:
        dataSetsToEvaluate = dataSetPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(dataSetPaths)
            if i in dataSetsToLoad
        ]

    for dataSetPath in dataSetsToEvaluate:
        # set file paths
        dataSetName = dataSetPath.split("/")[-2]
        resultFolderPath = savePath + dataSetName + "/"
        resultFilePath = resultFolderPath + resultFileName + ".pkl"

        results = {}
        results["dataSetPath"] = dataSetPath

        # select config
        configFilePath = "/evalConfigs/evalConfig" + "_" + dataSetName + ".json"

        # setup evalulation
        global eval
        pathToConfigFile = os.path.dirname(os.path.abspath(__file__)) + configFilePath
        eval = TrackingEvaluation(configFilePath=pathToConfigFile)

        # initialization
        if runOpt["localization"]:
            initializationResult = eval.runInitialization(
                dataSetPath,
                eval.config["frameForInitialization"],
                visualizeSOMResult=visOpt["som"],
                visualizeSOMIterations=visOpt["somIterations"],
                visualizeL1Result=visOpt["l1"],
                visualizeL1Iterations=visOpt["l1Iterations"],
                visualizeExtractionResult=visOpt["topologyExtraction"],
                visualizeIterations=visOpt["inverseKinematicsIterations"],
                visualizeCorresponanceEstimation=visOpt["correspondanceEstimation"],
                visualizeResult=visOpt["initializationResult"],
            )
            results["initializationResult"] = initializationResult
            if saveOpt["localizationResults"]:
                eval.saveResults(
                    folderPath=resultFolderPath,
                    generateUniqueID=False,
                    fileName=resultFileName,
                    results=results,
                    promtOnSave=False,
                    overwrite=True,
                )
        else:
            results["initializationResult"] = eval.loadResults(resultFilePath)[
                "initializationResult"
            ]
        # tracking
        if runOpt["tracking"]:
            results["trackingResults"] = []
            bdloModelParameters = eval.getModelParameters(
                dataSetPath=dataSetPath,
                numBodyNodes=eval.config["modelGeneration"]["numSegments"],
            )
            for registrationMethod in registrationsToRun:
                # determine initial and final frame for tracking
                initialFrame = eval.config["initialFrame"]
                if eval.config["finalFrame"] == -1:
                    finalFrame = eval.getNumImageSetsInDataSet(dataSetPath)
                else:
                    finalFrame = eval.config["finalFrame"]

                trackingResult = eval.runTracking(
                    dataSetPath=dataSetPath,
                    bdloModelParameters=bdloModelParameters,
                    method=registrationMethod,
                    startFrame=eval.config["initialFrame"],
                    endFrame=eval.config["finalFrame"],
                    frameStep=eval.config["frameStep"],
                    checkConvergence=False,
                    XInit=results["initializationResult"]["localization"]["XInit"],
                    B=results["initializationResult"]["localization"]["BInit"],
                    S=results["initializationResult"]["localization"]["SInit"],
                    qInit=results["initializationResult"]["localization"]["qInit"],
                    visualize=visOpt["trackingIterations"],
                )
                results["trackingResults"].append(trackingResult)
                if saveOpt["trackingResults"]:
                    eval.saveResults(
                        folderPath=resultFolderPath,
                        generateUniqueID=False,
                        fileName=resultFileName,
                        results=results,
                        promtOnSave=False,
                        overwrite=True,
                    )
        else:
            results["trackingResults"] = eval.loadResults(resultFilePath)[
                "trackingResults"
            ]

        # evaluate results
        if runOpt["evaluation"]:
            trackingEvaluationResults = evaluateTrackingResults(results)
            results["trackingEvaluationResults"] = trackingEvaluationResults
            if saveOpt["trackingEvaluationResults"]:
                eval.saveResults(
                    folderPath=resultFolderPath,
                    generateUniqueID=False,
                    fileName=resultFileName,
                    results=results,
                    promtOnSave=False,
                    overwrite=True,
                )
