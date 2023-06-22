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


def evaluateGraspingAccuracy(dataSetPath, frame, initializationResult):
    graspingAccuracyResult = {}

    # run different tracking algorithms
    trackingResult = eval.runTracking(
        dataSetPath=dataSetPath,
        method="cpd",
        startFrame=frame,
        endFrame=frame,
        checkConvergence=True,
        XInit=initializationResult["localization"]["XInit"],
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
    graspingAccuracyResult["trackinResult"] = trackingResult
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


if __name__ == "__main__":
    initializationFrame = eval.config["initialFrame"]
    evaluationFrame = initializationFrame
    # initialize on the first frame of the data set
    if runExperiment:
        if loadInitializationFromResult:
            initializationResult = eval.loadResults(resultFilePath)["initialization"]
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
                )
        # evaluate grasping accuracy for different algorithms
        graspingAccuracyResult = evaluateGraspingAccuracy(
            dataSetPath, evaluationFrame, initializationResult
        )
    # else:
    #     results = eval.loadResults(resultFilePath)
    #     eval.results = results

    # evaluate results
