import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import logging
import traceback

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
global runOpt
global eval

runExperiment = True
runOpt = {"localization": False, "tracking": True}
saveOpt = {
    "localizationResults": False,
    "trackingResults": False,
    "evaluationResults": False,
}
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

# path configurations
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig_partial.json"
)
savePath = "data/eval/graspingAccuracy/results/"
logFileName = "graspingAccuracy.log"
resultFileName = "result"

# configure logging
logFile = savePath + logFileName
logLevel = logging.INFO
logFormat = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logFile, level=logLevel, format=logFormat)

# script starts here
eval = GraspingAccuracyEvaluation(configFilePath=pathToConfigFile)


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
    graspingAccuracyResults = []

    # gather required informatio
    dataSetPath = experimentResults["dataSetPath"]

    # get grasping label positions
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)

    for trackingResult in experimentResults["trackingResults"]:
        graspingAccuracyResult = {}

        B = trackingResult["registrationResults"]["B"]
        S = trackingResult["registrationResults"]["S"]

        groundTruthGraspingPoses = []
        groundTruthGraspingPositions = []
        groundTruthGraspingRotationMatrices = []
        groundTruthGraspingAxes = []
        predictedGraspingPositions = []
        predictedGraspingAxes = []
        predictedFrames = []
        groundTruthFrames = []
        for i, graspingLocalCoordinate in enumerate(graspingLocalCoordinates):
            registrationResult = trackingResult["registrationResults"]["registrations"][
                i
            ]
            frame = registrationResult["frame"]
            predictedFrames.append(frame)
            T = registrationResult["T"]

            # ground truth grasping pose
            groundTruthFrame = frame + 1
            groundTruthFrames.append(groundTruthFrame)
            (
                groundTruthGraspingPose,
                groundTruthGraspingPosition,
                groundTruthGraspingRotationMatrix,
            ) = eval.loadGroundTruthGraspingPose(
                dataSetPath, groundTruthFrame
            )  # get ground truth from the follwing frame
            groundTruthGraspingPoses.append(groundTruthGraspingPose)
            groundTruthGraspingPositions.append(groundTruthGraspingPosition)
            groundTruthGraspingRotationMatrices.append(
                groundTruthGraspingRotationMatrix
            )
            groundTruthGraspingAxes.append(groundTruthGraspingRotationMatrix[:3, 0])
            # predict the estimated grasping pose from the registration result
            (
                predictedGraspingPosition,
                predictedGraspingAxis,
            ) = eval.predictGraspingPositionAndAxisFromRegistrationTargets(
                T=T, B=B, S=S, graspingLocalCoordinate=graspingLocalCoordinate
            )
            predictedGraspingPositions.append(predictedGraspingPosition)
            predictedGraspingAxes.append(predictedGraspingAxis)

        # eval grasping error
        (
            graspingPositionErrors,
            graspingAngularErrorsInRad,
            graspingAngularErrorsInGrad,
            projectedGraspingAngularErrorsOnXInRad,
            projectedGraspingAngularErrorsOnXInGrad,
            projectedGraspingAngularErrorsOnYInRad,
            projectedGraspingAngularErrorsOnYInGrad,
            projectedGraspingAngularErrorsOnZInRad,
            projectedGraspingAngularErrorsOnZInGrad,
        ) = eval.calculateGraspingAccuracyError(
            predictedGraspingPositions=predictedGraspingPositions,
            predictedGraspingAxes=predictedGraspingAxes,
            groundTruthGraspingPoses=groundTruthGraspingPoses,
        )

        # gather results
        graspingAccuracyResult["method"] = trackingResult["registrationMethod"]
        graspingAccuracyResult["dataSetPath"] = dataSetPath
        graspingAccuracyResult["predictedFrames"] = predictedFrames
        graspingAccuracyResult["groundTruthFrames"] = groundTruthFrames
        graspingAccuracyResult["trackingResult"] = trackingResult
        graspingAccuracyResult["graspingLocalCoordinates"] = graspingLocalCoordinates
        # grasping position eval results
        graspingAccuracyResult["graspingPositions"] = {}
        graspingAccuracyResult["graspingPositions"][
            "predicted"
        ] = predictedGraspingPositions
        graspingAccuracyResult["graspingPositions"][
            "groundTruth"
        ] = groundTruthGraspingPositions
        graspingAccuracyResult["graspingPositionErrors"] = graspingPositionErrors
        graspingAccuracyResult["graspingAxes"] = {
            "predicted": predictedGraspingAxes,
            "groundTruth": groundTruthGraspingAxes,
        }
        graspingAccuracyResult["graspingAngularErrors"] = {
            "rad": graspingAngularErrorsInRad,
            "grad": graspingAngularErrorsInGrad,
            "projected": {
                "X": {
                    "rad": projectedGraspingAngularErrorsOnXInRad,
                    "grad": projectedGraspingAngularErrorsOnXInGrad,
                },
                "Y": {
                    "rad": projectedGraspingAngularErrorsOnYInRad,
                    "grad": projectedGraspingAngularErrorsOnYInGrad,
                },
                "Z": {
                    "rad": projectedGraspingAngularErrorsOnZInRad,
                    "grad": projectedGraspingAngularErrorsOnZInGrad,
                },
            },
        }
        graspingAccuracyResult["gripperPoses"] = groundTruthGraspingPoses
        graspingAccuracyResults.append(graspingAccuracyResult)
    return graspingAccuracyResults


# def predictGraspingPoses_Backward():

if __name__ == "__main__":
    if eval.config["dataSetsToLoad"][0] == -1:
        dataSetPaths = eval.config["dataSetPaths"]
    else:
        dataSetPaths = [
            dataSetPath
            for i, dataSetPath in enumerate(eval.config["dataSetPaths"])
            if i in eval.config["dataSetsToLoad"]
        ]

    for dataSetPath in dataSetPaths:
        try:
            results = {}
            results["dataSetPath"] = dataSetPath
            # run experiments
            # set file paths
            dataSetName = dataSetPath.split("/")[-2]
            resultFolderPath = savePath + dataSetName + "/"
            resultFilePath = resultFolderPath + resultFileName + ".pkl"
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
                results["trackingResults"] = eval.loadResults(resultFilePath)[
                    "trackingResults"
                ]

            experimentResults = results
            graspingAccuracyResults = evaluateGraspingAccuracy(experimentResults)
            results["graspingAccuracyResults"] = graspingAccuracyResults
            # save evaluation results
            if saveOpt["evaluationResults"]:
                eval.saveResults(
                    folderPath=resultFolderPath,
                    generateUniqueID=False,
                    fileName=resultFileName,
                    results=results,
                    promtOnSave=False,
                    overwrite=True,
                )

            print('Evaluated data set: "{}"'.format(dataSetPath))
        except:
            traceback.print_exc()
            print('Failed on data set: "{}"'.format(dataSetPath))
            logging.info('Failed on data set: "{}"'.format(dataSetPath))
            raise
    print("Finised grasping accuracy evaluation".format(dataSetPath))
