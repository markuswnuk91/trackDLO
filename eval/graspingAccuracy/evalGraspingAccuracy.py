import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

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
        graspingPositionPredicted = branchInterpoationFun(sGrasp)
        graspingPositionsPredicted.append(graspingPositionPredicted)

    numMarkerGraspingPositions = len(graspingLocalCoordinates)
    # get grasping ground truth grasping poistions from robot measurement
    robotEETransformsGT = []
    robotEEPositionsGT = []
    robotEERotationMatricesGT = []
    for i in range(1, numMarkerGraspingPositions + 1):
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

    # gather results
    graspingAccuracyResult["graspingLocalCoordinates"] = graspingLocalCoordinates
    graspingAccuracyResult["graspingPositions"] = {}
    graspingAccuracyResult["graspingPositions"][
        "predicted"
    ] = graspingPositionsPredicted
    graspingAccuracyResult["graspingPositions"]["groundTruth"] = robotEEPositionsGT
    graspingAccuracyResult["graspingPositions"]["errors"] = graspingPositionErrors
    graspingAccuracyResult["graspingPositions"][
        "errorDistances"
    ] = gaspingPositionErrorDistances
    graspingAccuracyResult["trackinResult"] = trackingResult
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
