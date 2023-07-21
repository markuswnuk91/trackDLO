import sys
import os
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.evaluation import Evaluation

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global save
save = False
global vis
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
runExperiment = True

# setup evalulation class
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = Evaluation(configFilePath=pathToConfigFile)
# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetToLoad"]]
dataSetName = eval.config["dataSetPaths"][0].split("/")[-2]
resultFolderPath = eval.config["resultFolderPath"] + dataSetName + "/"
resultFileName = "result"
resultFilePath = resultFolderPath + resultFileName + ".pkl"

# setup results
eval.results = {
    "dataSetPath": dataSetPath,
    "dataSetName": dataSetName,
    "pathToConfigFile": pathToConfigFile,
    "evalConfig": eval.config,
}


def runExperiments(dataSetPath, framesIndices):
    results = []
    failedFrames = []
    failCounter = 0
    for frameIdx in framesIndices:
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
            plt.pause(0.01)
        except:
            failCounter += 1
            failedFrames.append(eval.getFilePath(frameIdx, dataSetPath))
    return results, failCounter, failedFrames


def evaluateExperiment():
    raise NotImplementedError


if __name__ == "__main__":
    # setup experiment

    if runExperiment:
        # run experiments
        numImagesInDataSet = eval.getNumImageSetsInDataSet(
            dataSetFolderPath=dataSetPath
        )
        frameIndices = list(range(0, numImagesInDataSet))
        # frameIndices = [0]
        results, numFailures, failedFrames = runExperiments(dataSetPath, frameIndices)
        eval.results["results"] = results
        eval.results["numFailures"] = numFailures
        eval.results["FailedFrames"] = failedFrames
    else:
        results = eval.loadResults(resultFilePath)

    # evaluate experiments

    # save results
    if save:
        eval.saveResults(
            folderPath=resultFolderPath,
            generateUniqueID=False,
            fileName=resultFileName,
            promtOnSave=False,
            overwrite=True,
        )
    print("Failures: {}".format(results["numFailures"]))
