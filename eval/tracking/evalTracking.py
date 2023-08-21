import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from warnings import warn

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
    "initializationResult": True,
    "trackingIterations": True,
}
saveOpt = {
    "localizationResults": False,
    "trackingResults": True,
    "saveRegistrationsAsImage": True,
    "evaluationResults": True,
}
registrationsToRun = [
    # "cpd",
    # "spr",
    # "kpr",
    "krcpd",
    # "krcpd4BDLO",
]  # cpd, spr, krcpd, krcpd4BDLO
dataSetsToLoad = [0]  # -1 to load all data sets

savePath = "data/eval/tracking/results/"
resultFileName = "result"
dataSetPaths = [
    "data/darus_data_download/data/20230524_171237_ManipulationSequences_mountedWireHarness_modelY/",
    "data/darus_data_download/data/20230524_161235_ManipulationSequences_mountedWireHarness_arena/",
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
    geometricErrorResult = {"mean": [], "accumulated": []}
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
        geometricErrorResult["accumulated"].append(geometricError)
        geometricErrorResult["mean"].append(meanGeometricError)
    return geometricErrorResult


def calculateReprojectionErrors(trackingMethodResult):
    reprojectionErrorResult = {}

    dataSetPath = trackingMethodResult["dataSetPath"]
    # get label local cooridnates
    markerLocalCoordinates = eval.getMarkerBranchLocalCoordinates(dataSetPath)
    # deteremine for which frames labels exist
    labelInfo = eval.loadLabelInfo(dataSetPath)
    labeledFrames = []
    labeledFramesFileNames = []
    groundTruthPixelCoordinatesForFrame = []
    missingLabelsForFrame = []
    for labelEntry in labelInfo:
        fileName = labelEntry["file_upload"].split("-")[-1]
        labeledFramesFileNames.append(fileName)
        filePath = dataSetPath + "data/" + fileName
        labeledFrames.append(eval.getFileIndexFromFileName(fileName, dataSetPath))
        (
            groundTruthPixelCoordinates,
            missingLabels,
        ) = eval.loadGroundTruthLabelPixelCoordinates(filePath)
        groundTruthPixelCoordinatesForFrame.append(groundTruthPixelCoordinates)
        missingLabelsForFrame.append(missingLabels)

    trackedFrames = trackingMethodResult["frames"]
    framesToEvaluate = list(set(labeledFrames) & set(trackedFrames))

    evaluatedFrames = []
    B = trackingMethodResult["B"]
    S = trackingMethodResult["S"]
    meanReprojectionErrorPerFrame = []
    reprojectionErrorsPerFrame = []
    predictedCoordinates2DPerFrame = []
    groundTruthCoordinates2DPerFrame = []
    correspondingTrackingMethodResults = []
    targetPositions3D = []
    evaluatedMarkers = []
    for frame in framesToEvaluate:
        groundTruthMarkerCoordinates2D = groundTruthPixelCoordinatesForFrame[
            labeledFrames.index(frame)
        ]
        # get tracking result cooresponding to labeld frame
        correspondingTrackingMethodResult = eval.findCorrespondingEntryFromKeyValuePair(
            trackingMethodResult["registrations"], "frame", frame
        )
        correspondingTrackingMethodResults.append(correspondingTrackingMethodResult)
        T = correspondingTrackingMethodResult["T"]
        targetPositions3D.append(T)
        predictedMarkerPositions3D = eval.interpolateRegistredTargets(
            T, B, S, markerLocalCoordinates
        )
        # reproject in 2D pixel coordinates
        predictedMarkerCoordinates2D = eval.reprojectFrom3DRobotBase(
            predictedMarkerPositions3D, dataSetPath
        )

        missingLabels = missingLabelsForFrame[labeledFrames.index(frame)]
        markersToEvaluate = list(
            set(list(range(1, len(predictedMarkerPositions3D) + 1)))
            - set(missingLabels)
        )
        evaluatedMarkers.append(markersToEvaluate)
        markerCoordinateIndices = np.array(markersToEvaluate) - 1
        reprojectionErrors = np.linalg.norm(
            predictedMarkerCoordinates2D[markerCoordinateIndices, :]
            - groundTruthMarkerCoordinates2D,
            axis=1,
        )
        meanReprojectionErrorPerFrame.append(np.mean(reprojectionErrors))
        predictedCoordinates2DPerFrame.append(predictedMarkerCoordinates2D)
        groundTruthCoordinates2DPerFrame.append(groundTruthMarkerCoordinates2D)
        reprojectionErrorsPerFrame.append(reprojectionErrors)
        evaluatedFrames.append(frame)
        reprojectionErrorResult["frames"] = framesToEvaluate
        reprojectionErrorResult["mean"] = meanReprojectionErrorPerFrame
        reprojectionErrorResult[
            "predictedMarkerCoordinates2D"
        ] = predictedCoordinates2DPerFrame
        reprojectionErrorResult[
            "groundTruthMarkerCoordinates2D"
        ] = groundTruthCoordinates2DPerFrame
        reprojectionErrorResult["reprojectionErrors"] = reprojectionErrorsPerFrame
        reprojectionErrorResult["targetPositions3D"] = targetPositions3D
        reprojectionErrorResult["B"] = B
        reprojectionErrorResult["S"] = S
        reprojectionErrorResult["evaluatedMarkers"] = evaluatedMarkers
    return reprojectionErrorResult


def calculateRuntimes(trackingMethodResult):
    runtimeResults = trackingMethodResult["runtimes"]
    runtimeResults["mean"] = np.mean(
        trackingMethodResult["runtimes"]["runtimesPerIteration"]
    )
    runtimeResults["std"] = np.std(
        trackingMethodResult["runtimes"]["runtimesPerIteration"]
    )
    numPointsPerIterations_Y = []
    numPointsPerIterations_X = []
    numCorrespondancesPerIteration = []
    for registrationResult in trackingMethodResult["registrations"]:
        numPoints_Y = len(registrationResult["Y"])
        numPointsPerIterations_Y.append(numPoints_Y)
        numPoints_X = len(registrationResult["X"])
        numPointsPerIterations_X.append(numPoints_X)
        numCorrespondances = numPoints_Y * numPoints_X
        numCorrespondancesPerIteration.append(numCorrespondances)
    runtimeResults["numPointsPerIteration"] = numPointsPerIterations_Y
    runtimeResults["numCorrespondancesPerIteration"] = numCorrespondancesPerIteration
    return runtimeResults


def evaluateTrackingResults(results):
    trackingEvaluationResults = {}
    dataSetPath = results["dataSetPath"]
    for method in results["trackingResults"]:
        trackingEvaluationResult = {}
        trackingMethodResult = results["trackingResults"][method]
        trackingEvaluationResult["trackingResult"] = results["trackingResults"][method]
        # tracking errors
        trackingErrors = calculateTrackingErrors(trackingMethodResult)
        trackingEvaluationResult["trackingErrors"] = trackingErrors
        # geometric errors
        geometricErrors = calculateGeometricErrors(trackingMethodResult)
        trackingEvaluationResult["geometricErrors"] = geometricErrors

        # reprojection errors
        # check if data set has labels
        hasLabels = eval.checkLabels(dataSetPath)
        if hasLabels:
            reprojectionErrors = calculateReprojectionErrors(trackingMethodResult)
            trackingEvaluationResult["reprojectionErrors"] = reprojectionErrors
            # plot reprojection errors
            model = eval.generateModel(
                modelParameters=results["trackingResults"][0]["modelParameters"],
            )
            adjacencyMatrix = model.getBodyNodeNodeAdjacencyMatrix()
            for i, evalFrame in enumerate(reprojectionErrors["frames"]):
                T = trackingMethodResult["registrations"][evalFrame]["T"]
                markerIndicesToEvaluate = (
                    np.array(reprojectionErrors["evaluatedMarkers"][i]) - 1
                )
                predictedMarkerCoordinates2D = reprojectionErrors[
                    "predictedMarkerCoordinates2D"
                ][i][markerIndicesToEvaluate, :]
                groundTruthMarkerCoordinates2D = reprojectionErrors[
                    "groundTruthMarkerCoordinates2D"
                ][i]
                evaluatedMarkers = reprojectionErrors["evaluatedMarkers"][i]
                eval.visualizeReprojectionError(
                    fileName=eval.getFileName(evalFrame, dataSetPath),
                    dataSetPath=dataSetPath,
                    positions3D=T,
                    adjacencyMatrix=adjacencyMatrix,
                    predictedMarkerCoordinates2D=predictedMarkerCoordinates2D,
                    groundTruthMarkerCoordinates2D=groundTruthMarkerCoordinates2D,
                    block=False,
                )
        else:
            warn(
                "No annotated ground truth labels found. Proceeding without calculating reprojection error."
            )
        # runtime
        runtimeResults = calculateRuntimes(trackingMethodResult)
        trackingEvaluationResult["runtimeResults"] = runtimeResults
        trackingEvaluationResults[method] = trackingEvaluationResult
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
            results["trackingResults"] = {}
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

                if saveOpt["saveRegistrationsAsImage"]:
                    registrationsSavePath = (
                        resultFolderPath + "registrations/" + registrationMethod
                    ) + "/"
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
                    savePath=registrationsSavePath,
                )
                results["trackingResults"][registrationMethod] = trackingResult
                # ensure methods are not overridden
                if saveOpt["trackingResults"]:
                    if os.path.exists(resultFilePath):
                        existingResults = eval.loadResults(resultFilePath)[
                            "trackingResults"
                        ]
                        for method in existingResults:
                            if method not in registrationsToRun:
                                results["trackingResults"][method] = existingResults[
                                    method
                                ]
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
            if saveOpt["evaluationResults"]:
                eval.saveResults(
                    folderPath=resultFolderPath,
                    generateUniqueID=False,
                    fileName=resultFileName,
                    results=results,
                    promtOnSave=False,
                    overwrite=True,
                )
