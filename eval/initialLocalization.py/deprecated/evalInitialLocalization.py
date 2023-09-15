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

configFile = "partial"  # sigleDLO, modelY, partial, arena
save = False
saveImgs = False
runExperiment = True  # if localization should be run or loaded from data
runEvaluation = True
runExperimentsForFrames = [1]  # options: -1 for all frames, else number of frames
vis = {
    "som": False,
    "somIterations": True,
    "l1": False,
    "l1Iterations": True,
    "extraction": True,
    "iterations": True,
    "correspondances": False,
    "initializationResult": False,
    "reprojectionErrorEvaluation": True,
    "reprojectionErrorTimeSeries": True,
}


# setup evalulation class
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__))
    + "/evalConfigs/evalConfig_"
    + configFile
    + ".json"
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

        # check if we have ground truth labels
        fileName = initializationResult["fileName"]
        labelsDict = eval.loadLabelInfo(initializationResult["dataSetPath"])
        labelInfo = eval.findCorrespondingLabelEntry(fileName, labelsDict)
        if labelInfo is not None:
            pass
        else:
            break
        # corresponding initialization result
        evaluationResult["initializationResult"] = initializationResult

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

        # runtime evaluation
        runTimeEvaluationResult = evaluateRuntime(initializationResult)
        evaluationResult["runtimeEvaluation"] = runTimeEvaluationResult

        evaluationResults.append(evaluationResult)
    return evaluationResults


def evaluateRuntime(initializationResult):
    """Gathers relevant runtime measurement in a result structure"""
    evalResult = {}

    # number of points in the data set
    evalResult["numberOfPoints"] = len(initializationResult["localization"]["Y"])

    # preprocessing
    evalResult["pointCloudProcessing"] = initializationResult["runtimes"][
        "pointCloudProcessing"
    ]
    evalResult["modelGeneration"] = initializationResult["runtimes"]["modelGeneration"]

    evalResult["preprocessing"] = (
        evalResult["pointCloudProcessing"] + evalResult["modelGeneration"]
    )

    # som
    # total
    evalResult["runtime_som"] = initializationResult["runtimes"]["topologyExtraction"][
        "som"
    ]["withoutVisualization"]
    # iterations
    evalResult["meanRuntimePerIteration_som"] = np.mean(
        initializationResult["runtimes"]["topologyExtraction"]["som"]["perIteration"]
    )
    evalResult["stdRuntimePerIteration_som"] = np.std(
        initializationResult["runtimes"]["topologyExtraction"]["som"]["perIteration"]
    )
    evalResult["numberOfIterations_som"] = len(
        initializationResult["runtimes"]["topologyExtraction"]["som"]["perIteration"]
    )

    # l1
    # total
    evalResult["runtime_l1"] = initializationResult["runtimes"]["topologyExtraction"][
        "l1"
    ]["withoutVisualization"]
    # iterations
    evalResult["meanRuntimePerIteration_l1"] = np.mean(
        initializationResult["runtimes"]["topologyExtraction"]["l1"]["perIteration"]
    )
    evalResult["stdRuntimePerIteration_l1"] = np.std(
        initializationResult["runtimes"]["topologyExtraction"]["l1"]["perIteration"]
    )
    evalResult["numberOfIterations_l1"] = len(
        initializationResult["runtimes"]["topologyExtraction"]["l1"]["perIteration"]
    )

    # mst-topologyReconstruction
    evalResult["runtime_mstTopologyReconstruction"] = initializationResult["runtimes"][
        "topologyExtraction"
    ]["topologyExtraction"]["topologyExtraction"]

    # correspondance estimation
    evalResult["runtime_correspondanceEstimation"] = initializationResult["runtimes"][
        "localization"
    ]["correspondanceEstimation"]

    # inverse kinematics
    evalResult["runtime_inverseKinematics"] = np.sum(
        initializationResult["runtimes"]["localization"]["inverseKinematicsIterations"]
    )
    evalResult["meanRuntimePerIteration_inverseKinematics"] = np.mean(
        initializationResult["runtimes"]["localization"]["inverseKinematicsIterations"]
    )
    evalResult["stdRuntimePerIteration_inverseKinematics"] = np.std(
        initializationResult["runtimes"]["localization"]["inverseKinematicsIterations"]
    )
    evalResult["runtime_total"] = (
        evalResult["preprocessing"]
        + evalResult["runtime_som"]
        + evalResult["runtime_l1"]
        + evalResult["runtime_mstTopologyReconstruction"]
        + evalResult["runtime_correspondanceEstimation"]
        + evalResult["runtime_inverseKinematics"]
    )
    return evalResult


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
    evalResult["q"] = initializationResult["localization"]["q"]
    evalResult["modelParameters"] = initializationResult["modelParameters"]
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
        modelParameters=reprojectionErrorResult["modelParameters"],
        q=reprojectionErrorResult["q"],
        markerBranchCoordinates=eval.getMarkerBranchLocalCoordinates(dataSetPath),
        groundTruthLabelCoordinates=reprojectionErrorResult[
            "groundTruthLabelCoordinates_2D"
        ],
        plotGrayScale=False,
        save=saveImgs,
        savePath=resultFolderPath,
        block=False,
    )


def tabularizeRuntimeResults(evaluationResults):
    runtimeResults = {
        "model": eval.getModelID(
            evaluationResults[0]["initializationResult"]["modelParameters"][
                "modelInfo"
            ]["name"]
        ),
        "n_config": len(evaluationResults),
        "t_preprocessing": 0.0,
        "t_l1_skel": 0.0,
        "t_som": 0.0,
        "t_mst": 0.0,
        "t_corresp": 0.0,
        "t_ik": 0.0,
        "t_total": 0.0,
    }

    runtimesPreprocessing = []
    runtimesL1 = []
    runtimesSOM = []
    runtimesMST = []
    runtimesCorrespEst = []
    runtimesIK = []

    for result in evaluationResults:
        runtimeResult = result["runtimeEvaluation"]
        runtimesPreprocessing.append(runtimeResult["preprocessing"])
        runtimesL1.append(runtimeResult["runtime_l1"])
        runtimesSOM.append(runtimeResult["runtime_som"])
        runtimesMST.append(runtimeResult["runtime_mstTopologyReconstruction"])
        runtimesCorrespEst.append(runtimeResult["runtime_correspondanceEstimation"])
        runtimesIK.append(runtimeResult["runtime_inverseKinematics"])

    runtimeResults["t_preprocessing"] = np.mean(runtimesPreprocessing)
    runtimeResults["t_l1_skel"] = np.mean(runtimesL1)
    runtimeResults["t_som"] = np.mean(runtimesSOM)
    runtimeResults["t_mst"] = np.mean(runtimesMST)
    runtimeResults["t_corresp"] = np.mean(runtimesCorrespEst)
    runtimeResults["t_ik"] = np.mean(runtimesIK)
    runtimeResults["t_total"] = (
        runtimeResults["t_preprocessing"]
        + runtimeResults["t_l1_skel"]
        + runtimeResults["t_som"]
        + runtimeResults["t_mst"]
        + runtimeResults["t_corresp"]
        + runtimeResults["t_ik"]
        + runtimeResults["t_total"]
    )

    latex_table_column = f"""
        ${runtimeResults['model']}$ & ${runtimeResults['n_config']}$ & ${runtimeResults['t_preprocessing']:.2f}$ & ${runtimeResults['t_l1_skel']:.2f}$ & ${runtimeResults['t_som']:.2f}$ & ${runtimeResults['t_mst']:.2f}$ & ${runtimeResults['t_corresp']:.2f}$ & ${runtimeResults['t_ik']:.2f}$ & $\\mathbf{{{runtimeResults['t_total']:.2f}}}$ \\\\
    """
    # meanTotalRuntimeForInitialization =
    print(latex_table_column)
    return runtimeResults


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
            frameIndices = runExperimentsForFrames
        # frameIndices = list(range(0, numImagesInDataSet))
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
    plt.show(block=True)
    tabularizeRuntimeResults(results["evaluationResults"])
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
