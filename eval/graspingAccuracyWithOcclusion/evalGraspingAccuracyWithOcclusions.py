import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.evaluation import Evaluation
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )

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
save = False
runExperiment = True
loadInitializationFromResult = True

# setup evalulation class
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

# setup results
eval.results = {
    "dataSetPath": dataSetPath,
    "dataSetName": dataSetName,
    "pathToConfigFile": pathToConfigFile,
    "evalConfig": eval.config,
}


def runExperiments(dataSetPath, initializationResult, startFrame):
    results = []
    numFramesInDataSet = eval.getNumImageSetsInDataSet(dataSetPath)
    framesToEvaluate = list(
        range(startFrame, numFramesInDataSet, eval.config["frameStep"])
    )
    bdloModelParameters = eval.getModelParameters(dataSetPath)
    bdloModel = eval.generateModel(bdloModelParameters)

    for frame in framesToEvaluate:
        graspingAccuracyResults = determineGraspingAccuracyResults(
            dataSetPath,
            frame,
            initializationResult,
            registrationMethods=eval.config["registrationMethodsToEvaluate"],
            bdloModel=bdloModel,
        )
        results.append(graspingAccuracyResults)
    return results


def determineGraspingAccuracyResults(
    dataSetPath, frame, initializationResult, registrationMethods, bdloModel
):
    graspingAccuracyResults = {}
    for registrationMethod in registrationMethods:
        graspingAccuracyResult = determineGraspingAccuracy(
            dataSetPath, frame, initializationResult, registrationMethod, bdloModel
        )
        graspingAccuracyResults[registrationMethod] = graspingAccuracyResult
    return graspingAccuracyResult


def determineGraspingAccuracy(
    dataSetPath, frame, initializationResult, registrationMethod, bdloModel
):
    graspingAccuracyResult = {}
    # run different tracking algorithms
    trackingResult = eval.runTracking(
        dataSetPath=dataSetPath,
        model=bdloModel,
        method=registrationMethod,
        startFrame=frame,
        endFrame=frame,
        checkConvergence=False,
        XInit=initializationResult["localization"]["XInit"],
        B=initializationResult["localization"]["BInit"],
        S=initializationResult["localization"]["SInit"],
        qInit=initializationResult["localization"]["qInit"],
    )

    # get grasping positions from desciption
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)
    # predict the grasping positions from the registration result
    graspingPositionsPredicted = []
    graspingAxesPredicted = []
    for graspingLocalCoordinate in graspingLocalCoordinates:
        correspondingIndices = [
            index
            for index, value in enumerate(initializationResult["localization"]["BInit"])
            if value
            == graspingLocalCoordinate[0]
            - 1  # account for branch indexing starting @ 1 in model desciption
        ]
        TCorresponding = trackingResult["registrations"][0]["T"][correspondingIndices]
        sCorresponding = np.array(initializationResult["localization"]["SInit"])[
            correspondingIndices
        ]
        sSortedIndices = np.argsort(sCorresponding)
        TSorted = TCorresponding[sSortedIndices]
        sSorted = sCorresponding[sSortedIndices]
        sGrasp = graspingLocalCoordinate[1]
        branchInterpoationFun = interp1d(sSorted, TSorted.T)
        # compute the predicted positons
        graspingPositionPredicted = branchInterpoationFun(sGrasp)
        graspingPositionsPredicted.append(graspingPositionPredicted)
        # compute the predicted axis
        # get the local coordniate of the closest point to the grasping position
        sNext = min(sSorted, key=lambda s: abs(s - sGrasp))
        graspingAxis = branchInterpoationFun(sNext) - branchInterpoationFun(sGrasp)
        if sNext < sGrasp:
            # revert direction if the next postition is before the grasp position
            graspingAxis = -graspingAxis
        # normalize axis
        graspingAxis = 1 / np.linalg.norm(graspingAxis) * graspingAxis
        graspingAxesPredicted.append(graspingAxis)

    numGroundTruthGraspingPositions = len(graspingLocalCoordinates)
    # get grasping ground truth grasping poistions from robot measurement
    robotEETransformsGT = []
    robotEEPositionsGT = []
    robotEERotationMatricesGT = []
    for i in range(1, numGroundTruthGraspingPositions + 1):
        (
            robotEETransformGT,
            robotEEPositionGT,
            robotEERotationMatrixGT,
        ) = eval.loadGroundTruthGraspingPositions(dataSetPath, i)
        robotEETransformsGT.append(robotEETransformGT)
        robotEEPositionsGT.append(robotEEPositionGT)
        robotEERotationMatricesGT.append(robotEERotationMatrixGT)

    # compare registered positions to ground truth positions
    graspingPositionErrors = np.array(robotEEPositionsGT) - np.array(
        graspingPositionsPredicted
    )
    gaspingPositionErrorDistances = np.linalg.norm(graspingPositionErrors, axis=1)

    # compare registrered angle to ground truth angle
    graspingAngularErrors_grad = []
    graspingAngularErrors_rad = []
    graspingAngularErrorsProjected_X_rad = []
    graspingAngularErrorsProjected_X_grad = []
    graspingAngularErrorsProjected_Y_rad = []
    graspingAngularErrorsProjected_Y_grad = []
    graspingAngularErrorsProjected_Z_rad = []
    graspingAngularErrorsProjected_Z_grad = []
    graspingAxesGT = []
    for i in range(0, numGroundTruthGraspingPositions):
        robotGripperAxis_X = robotEERotationMatricesGT[i][:, 0]
        robotGripperAxis_Y = robotEERotationMatricesGT[i][:, 1]
        robotGripperAxis_Z = robotEERotationMatricesGT[i][:, 2]
        graspingAxesGT.append(robotGripperAxis_X)
        # compute the angular error as deviation between predicted and measured grasping axis
        dotProduct = np.dot(robotGripperAxis_X, graspingAxesPredicted[i])
        # aligtn the direction if direction is inverted
        if dotProduct < 0:
            dotProduct = -dotProduct
        graspingAngularErrorInRad = np.arccos(dotProduct)
        graspingAngularErrorInDegree = np.degrees(graspingAngularErrorInRad)
        graspingAngularErrors_rad.append(graspingAngularErrorInRad)
        graspingAngularErrors_grad.append(graspingAngularErrorInDegree)
        # project angular error onto the individual gripper axes
        rotAxis = np.cross(robotGripperAxis_X, graspingAxesPredicted[i])
        rotAxisNorm = 1 / np.linalg.norm(rotAxis) * rotAxis
        rotVec = graspingAngularErrorInDegree * rotAxisNorm
        r = R.from_rotvec(rotVec, degrees=True)
        # X
        projectedError_X = np.dot(r.as_rotvec(), robotGripperAxis_X)
        graspingAngularErrorInRad_X = np.linalg.norm(projectedError_X)
        graspingAngularErrorInDegree_X = np.degrees(graspingAngularErrorInRad_X)
        graspingAngularErrorsProjected_X_grad.append(graspingAngularErrorInDegree_X)
        graspingAngularErrorsProjected_X_rad.append(graspingAngularErrorInRad_X)
        # Y
        projectedError_Y = np.dot(r.as_rotvec(), robotGripperAxis_Y)
        graspingAngularErrorInRad_Y = np.linalg.norm(projectedError_Y)
        graspingAngularErrorInDegree_Y = np.degrees(graspingAngularErrorInRad_Y)
        graspingAngularErrorsProjected_Y_grad.append(graspingAngularErrorInDegree_Y)
        graspingAngularErrorsProjected_Y_rad.append(graspingAngularErrorInDegree_Y)
        # Z
        projectedError_Z = np.dot(r.as_rotvec(), robotGripperAxis_Z)
        graspingAngularErrorInRad_Z = np.linalg.norm(projectedError_Z)
        graspingAngularErrorInDegree_Z = np.degrees(graspingAngularErrorInRad_Z)
        graspingAngularErrorsProjected_Z_grad.append(graspingAngularErrorInDegree_Z)
        graspingAngularErrorsProjected_Z_rad.append(graspingAngularErrorInRad_Z)
    # gather results
    graspingAccuracyResult["method"] = registrationMethod
    graspingAccuracyResult["dataSetPath"] = dataSetPath
    graspingAccuracyResult["frame"] = frame
    graspingAccuracyResult["fileName"] = eval.getFileName(frame, dataSetPath)
    graspingAccuracyResult["trackingResult"] = trackingResult
    graspingAccuracyResult["graspingLocalCoordinates"] = graspingLocalCoordinates
    # grasping position eval results
    graspingAccuracyResult["graspingPositions"] = {}
    graspingAccuracyResult["graspingPositions"][
        "predicted"
    ] = graspingPositionsPredicted
    graspingAccuracyResult["graspingPositions"]["groundTruth"] = robotEEPositionsGT
    graspingAccuracyResult["graspingPositionErrors"] = {
        "errorVectors": graspingPositionErrors,
        "euclideanDistances": gaspingPositionErrorDistances,
    }
    graspingAccuracyResult["graspingAxes"] = {
        "predicted": graspingAxesPredicted,
        "groundTruth": graspingAxesGT,
    }
    graspingAccuracyResult["graspingAngularErrors"] = {
        "rad": graspingAngularErrors_rad,
        "grad": graspingAngularErrors_grad,
        "projected": {
            "X": {
                "rad": graspingAngularErrorsProjected_X_rad,
                "grad": graspingAngularErrorsProjected_X_grad,
            },
            "Y": {
                "rad": graspingAngularErrorsProjected_Y_rad,
                "grad": graspingAngularErrorsProjected_Y_grad,
            },
            "Z": {
                "rad": graspingAngularErrorsProjected_Z_rad,
                "grad": graspingAngularErrorsProjected_Z_grad,
            },
        },
    }
    graspingAccuracyResult["gripperPoses"] = robotEETransformsGT
    return graspingAccuracyResult


def evaluateGraspingAccuracyResult(graspingAccuracyResult):
    # plot results

    # bar diagram of linear grasping error for each grasping position

    # bar diagram of angular grasping error for each grasping position

    raise NotImplementedError


if __name__ == "__main__":
    initializationFrame = eval.config["frameForInitialization"]
    evaluationFrame = initializationFrame
    # initialize on the first frame of the data set
    if runExperiment:
        if loadInitializationFromResult:
            initializationResult = eval.loadResults(resultFilePath)["initialization"]
            eval.results["initialization"] = initializationResult
        else:
            initializationResult = eval.runInitialization(
                dataSetPath, initializationFrame, visualize=False
            )
            eval.results["initialization"] = initializationResult
            if save:
                eval.saveResults(
                    folderPath=resultFolderPath,
                    generateUniqueID=False,
                    fileName=resultFileName,
                    promtOnSave=True,
                )
        # evaluate grasping accuracy for different algorithms
        eval.results["graspingAccuracyResults"] = []
        results = runExperiments(
            dataSetPath,
            initializationResult,
            eval.config["frameForStartingExperiments"],
        )
        eval.results["graspingAccuracyResults"] = results
        if save:
            eval.saveResults(
                folderPath=resultFolderPath,
                generateUniqueID=False,
                fileName=resultFileName,
                promtOnSave=False,
                overwrite=True,
            )
    else:
        results = eval.loadResults(resultFilePath)

    # evaluate results
    evaluateGraspingAccuracyResult()
