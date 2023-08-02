import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
except:
    print("Imports for Grasping Accuracy failed.")
    raise

# script control parameters
global vis
global saveOpt
global runOptx
runOpt = {"localization": False, "tracking": False}
saveOpt = {
    "localizationResults": False,
    "trackingResults": True,
    "evaluationResults": False,
}
runExperiment = True
loadInitializationFromResult = True

# script starts here
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = GraspingAccuracyEvaluation(configFilePath=pathToConfigFile)

# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetToLoad"]]
dataSetName = eval.config["dataSetPaths"][0].split("/")[-2]
resultFolderPath = "data/eval/graspingAccuracy/" + dataSetName + "/"
resultFileName = "result"
resultFilePath = resultFolderPath + resultFileName + ".pkl"
vis = {
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


def runExperiments():
    for dataSetPath in eval.config["dataSetPaths"]:
        runExperiment(dataSetPath)
    return


def runInitialization(dataSetPath):
    # initialization
    frameIdx = 0  # run intialization on fist frame of the dataset
    initializationResult = eval.runInitialization(
        dataSetPath,
        frameIdx,
        visualizeSOMResult=vis["som"],
        visualizeSOMIterations=vis["somIterations"],
        visualizeL1Result=vis["l1"],
        visualizeL1Iterations=vis["l1Iterations"],
        visualizeExtractionResult=vis["topologyExtraction"],
        visualizeIterations=vis["inverseKinematicsIterations"],
        visualizeCorresponanceEstimation=vis["correspondanceEstimation"],
        visualizeResult=vis["initializationResult"],
    )
    return initializationResult


def trackConfigurations(dataSetPath, initializationResult, registrationMethod):
    # get grasping label information
    graspingLabelsCoordinatesList = eval.loadGraspingLocalCoordinates(dataSetPath)

    bdloModelParameters = eval.getModelParameters(
        dataSetPath=dataSetPath,
        numBodyNodes=eval.config["modelGeneration"]["numSegments"],
    )
    frameIdx = 0
    trackingResults = []
    trackingResult = eval.runTracking(
        dataSetPath=dataSetPath,
        bdloModelParameters=bdloModelParameters,
        method=registrationMethod,
        startFrame=frameIdx,
        frameStep=3,  # skip frames with robot occlusion
        checkConvergence=False,
        XInit=initializationResult["localization"]["XInit"],
        B=initializationResult["localization"]["BInit"],
        S=initializationResult["localization"]["SInit"],
        qInit=initializationResult["localization"]["qInit"],
        visualize=vis["trackingIterations"],
    )
    result = {
        "dataSetPath": dataSetPath,
        "initializationResult": initializationResult,
        "registrationMethod": registrationMethod,
        "registrationResults": trackingResult,
    }
    return result
    # for grasping pose in dataSet
    # track configuration

    # predict grasping pose

    # predict release pose
    return graspingPosePredictionResult


# def predictGraspingPose(dataSetPath, previousConfiguration):
def evaluateGraspingAccuracy(experimentResults):
    graspingAccuracyResults = {}

    # gather required informatio
    dataSetPath = experimentResults["dataSetPath"]

    # get grasping label positions
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)

    for trackingResult in experimentResults["trackingResults"]:
        groundTruthGraspingPoses = []
        for i, registrationResult in enumerate(trackingResult["registrationResults"]):
            correspondingGraspingPoseFileIndex = i + 1

            # groud truth grasping pose
            (
                groundTruthGraspingPose,
                groundTruthGraspingPosition,
                groundTruthGraspingRotMat,
            ) = eval.loadGroundTruthGraspingPose(
                dataSetPath, correspondingGraspingPoseFileIndex
            )
            groundTruthGraspingPoses.append(groundTruthGraspingPose)

            # predict the estimated grasping pose from the registration result

    return graspingAccuracyResults


# def predictGraspingPoses_Backward():

if __name__ == "__main__":
    results = {}
    results["dataSetPath"] = dataSetPath
    # run experiments
    if runOpt["localization"]:
        initializationResult = runInitialization(dataSetPath)
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
    if runOpt["tracking"]:
        results["trackingResults"] = []
        for registrationMethod in eval.config["registrationMethodsToEvaluate"]:
            trackingResult = trackConfigurations(
                dataSetPath, results["initializationResult"], registrationMethod
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
        results["trackingResults"] = eval.loadResults(resultFilePath)["trackingResults"]

    experimentResults = results
    graspingAccuracyResults = evaluateGraspingAccuracy(experimentResults)
    # run evaluation
    # evaluationResult = evaluateGraspingPose

    # save evaluation results

    print("Done.")
