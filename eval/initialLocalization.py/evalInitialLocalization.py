import sys
import os
import matplotlib.pyplot as plt

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
runExperimentsForFrames = 1  # options: -1 for all frames, else nuber of frames
vis = {
    "som": True,
    "somIterations": True,
    "l1": True,
    "l1Iterations": True,
    "extraction": True,
    "iterations": True,
    "correspondances": False,
    "initializationResult": True,
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
            failedFrames.append(eval.getFilePath(frameIdx, dataSetPath))
    return results, failCounter, failedFrames


def evaluateExperiment(initializationResult):
    # get the file name corresponding to this result
    dataSetFilePath = initializationResult["filePath"]
    dataSetPath = initializationResult["dataSetPath"]
    # load the corresponding ground trtuh label coordinates
    groundTruthLabelCoordinates = eval.loadGroundTruthLabelPixelCoordinates(
        dataSetFilePath
    )

    # get the local branch coordinates for the labels

    # evaluate reprojection error
    model = eval.generateModel(initializationResult["modelParameters"])
    modelInfo = eval.dataHandler.loadModelParameters("model.json", dataSetPath)
    labelBranchCoordinates = (modelInfo["labels"][0]["branch"],)
    # reprojectedCoordinates =
    # project solution in 2D space
    # projectedCoordinates =

    print(groundTruthLabelCoordinates)


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
        results["initializationResult"] = initializationResults
        results["numFailures"] = numFailures
        results["FailedFrames"] = failedFrames
    else:
        results = eval.loadResults(resultFilePath)

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

    # evaluate experiments
    evaluateExperiment(results["initializationResult"][0])
