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
    from src.visualization.plot2D import *

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
    "reprojectionErrorEvaluation": True,
    "reprojectionErrorTimeSeries": True,
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

        # reprojection error time series evaluation
        reprojectionErrorTimeSeriesResult = evaluateReprojectionErrorTimeSeries(
            initializationResult
        )
        evaluationResult[
            "reprojectionErrorTimeSeriesEvaluation"
        ] = reprojectionErrorTimeSeriesResult

        evaluationResults.append(evaluationResult)
    return evaluationResults


def evaluateReprojectionErrorTimeSeries(initializationResult):
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

    reprojectionErrorTimeSeries = []
    meanReprojectionErrorTimeSeries = []
    markerCoordinates3DTimeSeries = []
    markerCoordinates2DTimeSeries = []
    for q in initializationResult["localization"]["qLog"]:
        markerCoordinates_3D = model.computeForwardKinematicsFromBranchLocalCoordinates(
            q=q,
            branchLocalCoordinates=markerBranchLocalCoordinates,
        )
        # reproject in 2D pixel coordinates
        markerCoordinates_2D = eval.reprojectFrom3DRobotBase(
            markerCoordinates_3D, dataSetPath
        )
        # evaluate reprojection error
        reprojectionErrors = groundTruthLabelCoordinates_2D - markerCoordinates_2D
        meanReprojectionError = np.mean(np.linalg.norm(reprojectionErrors, axis=1))

        reprojectionErrorTimeSeries.append(reprojectionErrors)
        meanReprojectionErrorTimeSeries.append(meanReprojectionError)
        markerCoordinates3DTimeSeries.append(markerCoordinates_3D)
        markerCoordinates2DTimeSeries.append(markerCoordinates2DTimeSeries)

    evalResult["filePath"] = dataSetFilePath
    evalResult["dataSetPath"] = dataSetPath
    evalResult["groundTruthLabelCoordinates2D"] = groundTruthLabelCoordinates_2D
    evalResult["markerBranchLocalCoordinates"] = markerBranchLocalCoordinates
    evalResult["markerCoordinates3DTimeSeries"] = markerCoordinates3DTimeSeries
    evalResult["markerCoordinates2DTimeSeries"] = markerCoordinates2DTimeSeries
    evalResult["reprojectionErrorsTimeSeries"] = reprojectionErrorTimeSeries
    evalResult["meanReprojectionErrorTimeSeries"] = meanReprojectionErrorTimeSeries
    evalResult["initializationResult"] = initializationResult
    return evalResult


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

    eval.visualizeReprojectionError(
        fileName=fileName,
        dataSetPath=reprojectionErrorResult["dataSetPath"],
        modelParameters=reprojectionErrorResult["initializationResult"][
            "modelParameters"
        ],
        q=reprojectionErrorResult["initializationResult"]["localization"]["q"],
        markerBranchCoordinates=eval.getMarkerBranchLocalCoordinates(dataSetPath),
        groundTruthLabelCoordinates=reprojectionErrorResult[
            "groundTruthLabelCoordinates_2D"
        ],
        plotGrayScale=False,
        save=True,
        savePath=resultFolderPath,
        block=False,
    )


def plotReprojectionErrorTimeSeries(results):
    # collect all time series data in a list
    reprojectionErrorTimeSeriesData = [
        result["reprojectionErrorTimeSeriesEvaluation"][
            "meanReprojectionErrorTimeSeries"
        ]
        for result in results["evaluationResults"]
    ]
    eval.plotTimeSeries(reprojectionErrorTimeSeriesData)


# fig = plt.figure()
# ax = fig.add_subplot()
# numResults = len(reprojectionErrorTimeSeriesResults)
# for i, reprojectionErrorTimeSeriesResult in enumerate(
#     reprojectionErrorTimeSeriesResults
# ):
#     meanReprojectionErrorTimeSeries = reprojectionErrorTimeSeriesResult[
#         "meanReprojectionErrorTimeSeries"
#     ]
#     X = np.array(list(range(len(meanReprojectionErrorTimeSeries))))
#     Y = meanReprojectionErrorTimeSeries
#     color = eval.colorMaps["viridis"].to_rgba(i / (numResults - 1))
#     ax.plot(X, Y, color=color)
# plt.show(block=True)


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
    if vis["reprojectionErrorEvaluation"]:
        for evaluationResult in results["evaluationResults"]:
            visualizeReprojectionError(evaluationResult["reprojectionErrorEvaluation"])

    if vis["reprojectionErrorTimeSeries"]:
        plotReprojectionErrorTimeSeries(results)

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
