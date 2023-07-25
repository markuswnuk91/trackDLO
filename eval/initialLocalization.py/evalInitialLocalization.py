import sys
import os
import matplotlib.pyplot as plt
import cv2

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global save
global vis
global eval

save = False
runExperiment = False  # if localization should be run or loaded from data
runEvaluation = True
runExperimentsForFrames = 3  # options: -1 for all frames, else nuber of frames
vis = {
    "som": False,
    "somIterations": True,
    "l1": False,
    "l1Iterations": True,
    "extraction": False,
    "iterations": True,
    "correspondances": False,
    "initializationResult": False,
}


# setup evalulation class
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = InitialLocalizationEvaluation(configFilePath=pathToConfigFile)
# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetToLoad"]]
dataSetName = eval.config["dataSetPaths"][0].split("/")[-2]
resultFolderPath = eval.config["resultFolderPath"] + dataSetName + "/"
resultFileName = "result"
resultFilePath = resultFolderPath + resultFileName + ".pkl"


def runExperiments(dataSetPath, frameIndices):
    results = []
    failedFrames = []
    failCounter = 0
    for frameIdx in frameIndices:
        try:
            initializationResult = eval.runInitialization(
                dataSetPath,
                frameIdx,
                visualizeSOMResult=vis["som"],
                visualizeSOMIterations=vis["somIterations"],
                visualizeL1Result=vis["l1"],
                visualizeL1Iterations=vis["l1Iterations"],
                visualizeExtractionResult=vis["extraction"],
                visualizeIterations=vis["iterations"],
                visualizeResult=vis["initializationResult"],
            )
            results.append(initializationResult)
            plt.show(block=False)
            plt.pause(0.01)
        except:
            failCounter += 1
            failedFilePath = eval.getFilePath(frameIdx, dataSetPath)
            failedFrames.append(failedFilePath)
            print("Failed on dataset: {}".format(failedFilePath))
    return results, failCounter, failedFrames


def evaluateExperiments(initializationResults):
    evaluationResults = []
    for initializationResult in initializationResults:
        evaluationResult = {}

        # reprojection error evaluation
        reprojectionErrorResult = evaluateReprojectionError(initializationResult)
        evaluationResult["reprojectionErrorEvaluation"] = reprojectionErrorResult
        evaluationResults.append(evaluationResult)

    return evaluationResults


def evaluateReprojectionError(initializationResult):
    evalResult = {}
    # get the file name corresponding to this result
    dataSetFilePath = initializationResult["filePath"]
    dataSetPath = initializationResult["dataSetPath"]
    # load the corresponding ground trtuh label coordinates
    groundTruthLabelCoordinates_2D = eval.loadGroundTruthLabelPixelCoordinates(
        dataSetFilePath
    )
    # get the local branch coordinates
    markerBranchLocalCoordinates = eval.getMarkerBranchLocalCoordinates(dataSetPath)
    # get predicted 3D coordinates
    model = eval.generateModel(initializationResult["modelParameters"])
    modelInfo = eval.dataHandler.loadModelParameters("model.json", dataSetPath)
    markerCoordinates_3D = model.computeForwardKinematicsFromBranchLocalCoordinates(
        q=initializationResult["localization"]["q"],
        branchLocalCoordinates=markerBranchLocalCoordinates,
    )
    # reproject in 2D pixel coordinates
    markerCoordinates_2D = eval.reprojectFrom3DRobotBase(
        markerCoordinates_3D, dataSetPath
    )
    # evaluate reprojection error
    reprojectionErrors = groundTruthLabelCoordinates_2D - markerCoordinates_2D
    meanReprojectionError = np.mean(np.linalg.norm(reprojectionErrors, axis=1))

    evalResult["filePath"] = dataSetFilePath
    evalResult["dataSetPath"] = dataSetPath
    evalResult["groundTruthLabelCoordinates_2D"] = groundTruthLabelCoordinates_2D
    evalResult["markerBranchLocalCoordinates"] = markerBranchLocalCoordinates
    evalResult["markerCoordinates_3D"] = markerCoordinates_3D
    evalResult["markerCoordinates_2D"] = markerCoordinates_2D
    evalResult["reprojectionErrors"] = reprojectionErrors
    evalResult["meanReprojectionError"] = meanReprojectionError
    evalResult["initializationResult"] = initializationResult
    return evalResult


def visualizeReprojectionError(reprojectionErrorResult):
    # get path infos
    filePath = reprojectionErrorResult["filePath"]
    dataSetPath = reprojectionErrorResult["dataSetPath"]
    fileName = eval.getFileIdentifierFromFilePath(filePath)

    # load image
    rgbImg = eval.getDataSet(fileName, dataSetPath)[0]

    # plot model
    model = eval.generateModel(
        reprojectionErrorResult["initializationResult"]["modelParameters"]
    )
    modelInfo = eval.dataHandler.loadModelParameters("model.json", dataSetPath)

    jointPositions3D = np.concatenate(
        model.getAdjacentPointPairs(
            q=reprojectionErrorResult["initializationResult"]["localization"]["q"]
        ),
        axis=0,
    )
    jointPositions2D = eval.reprojectFrom3DRobotBase(jointPositions3D, dataSetPath)

    markerCoordinates_2D = reprojectionErrorResult["markerCoordinates_2D"]
    groundTruthLabelCoordinates_2D = reprojectionErrorResult[
        "groundTruthLabelCoordinates_2D"
    ]
    # draw image
    i = 0
    while i <= len(jointPositions2D[:, 0]) - 1:
        cv2.line(
            rgbImg,
            (jointPositions2D[:, 0][i], jointPositions2D[:, 1][i]),
            (jointPositions2D[:, 0][i + 1], jointPositions2D[:, 1][i + 1]),
            (0, 255, 0),
            5,
        )
        i += 2
    for markerPosition in markerCoordinates_2D:
        cv2.circle(rgbImg, markerPosition, 5, [255, 0, 0], -1)
    for labelPosition in groundTruthLabelCoordinates_2D:
        cv2.circle(rgbImg, labelPosition, 5, [0, 0, 255], -1)
    cv2.imshow("RGB image", cv2.resize(rgbImg, None, fx=0.8, fy=0.8))
    key = cv2.waitKey()
    print("Done")


if __name__ == "__main__":
    # setup experiment

    if runExperiment:
        # run experiments
        if runExperimentsForFrames == -1:
            numImagesInDataSet = eval.getNumImageSetsInDataSet(
                dataSetFolderPath=dataSetPath
            )
        else:
            numImagesInDataSet = runExperimentsForFrames
        frameIndices = list(range(0, numImagesInDataSet))
        # frameIndices = [0]
        initializationResults, numFailures, failedFrames = runExperiments(
            dataSetPath, frameIndices
        )

        # setup results
        results = {
            "dataSetPath": dataSetPath,
            "dataSetName": dataSetName,
            "pathToConfigFile": pathToConfigFile,
            "evalConfig": eval.config,
        }
        results["initializationResults"] = initializationResults
        results["numFailures"] = numFailures
        results["FailedFrames"] = failedFrames
        # save results
        if save:
            eval.saveResults(
                folderPath=resultFolderPath,
                results=results,
                generateUniqueID=False,
                fileName=resultFileName,
                promtOnSave=False,
                overwrite=True,
            )
    else:
        results = eval.loadResults(resultFilePath)

    # evaluate experiments
    if runEvaluation:
        results["evaluationResults"] = evaluateExperiments(
            results["initializationResults"]
        )

    visualizeReprojectionError(
        results["evaluationResults"][0]["reprojectionErrorEvaluation"]
    )

    # save results
    if save:
        eval.saveResults(
            folderPath=resultFolderPath,
            results=results,
            generateUniqueID=False,
            fileName=resultFileName,
            promtOnSave=False,
            overwrite=True,
        )
