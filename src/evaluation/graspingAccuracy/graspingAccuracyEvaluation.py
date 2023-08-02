import sys
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/graspingAccuracy", ""))
    from src.evaluation.evaluation import Evaluation
except:
    print("Imports for class TrackingEvaluation failed.")
    raise


class GraspingAccuracyEvaluation(Evaluation):
    def __init__(self, configFilePath, *args, **kwargs):
        super().__init__(configFilePath, *args, **kwargs)

    def loadGraspingPositionDescription(self, dataSetFolderPath, fileName=None):
        if fileName is None:
            fileName = "graspingPositions.json"
        filePath = dataSetFolderPath + fileName
        graspingPositionDesciption = self.dataHandler.loadFromJson(filePath)
        return graspingPositionDesciption

    def loadGraspingLocalCoordinates(self, dataSetFolderPath, fileName=None):
        graspingLocalCoordinates = []
        graspingPositionDesciption = self.loadGraspingPositionDescription(
            dataSetFolderPath, fileName=None
        )
        for graspingPosition in graspingPositionDesciption["graspingPositions"]:
            graspingLocalCoordinates.append(
                (graspingPosition["branch"] - 1, graspingPosition["s"])
            )
        return graspingLocalCoordinates

    def loadGroundTruthGraspingPose(self, dataSetFolderPath, fileNumber):
        filePath = self.dataHandler.getFilePath(
            fileNumber, dataSetFolderPath, fileType="json"
        )
        robotState = self.dataHandler.loadFromJson(filePath)
        transform_BaseToEE = np.reshape(
            np.array(robotState["O_T_EE"]), (4, -1), order="F"
        )
        robotEEPosition = transform_BaseToEE[:3, 3]
        robotEERotationMatrix = transform_BaseToEE[:3, :3]
        return transform_BaseToEE, robotEEPosition, robotEERotationMatrix

    def predictGraspingPositionAndAxisFromRegistrationTargets(
        self, T, B, S, graspingLocalCoordinate: tuple
    ):
        """Predict the grasping position and axis from registered target positions

        Args:
            T (np.array): Registered target positions
            B (list): list of branch indices the targets in T correspond to, branch indeces in range form 0,...,K
            S (list): list of local coordinates the targets in T correspond to.
            graspingLocalCoordinates (tuple): local branch coordinates as (b,s) where b is the branch index and s is the local coordiante along the branch. Branch indices from 0,...,K.
        """
        correspondingIndices = [
            index
            for index, value in enumerate(B)
            if value == graspingLocalCoordinate[0]
        ]
        TCorresponding = T[correspondingIndices, :]
        sCorresponding = np.array(S)[correspondingIndices]

        # interpolate target positions to get grasping pose
        sSortedIndices = np.argsort(sCorresponding)
        TSorted = TCorresponding[sSortedIndices]
        sSorted = sCorresponding[sSortedIndices]
        sGrasp = graspingLocalCoordinate[1]
        branchInterpoationFun = interp1d(sSorted, TSorted.T)
        predictedGraspingPosition = branchInterpoationFun(sGrasp)

        # calculate grasping axis
        sNext = min(sSorted, key=lambda s: abs(s - sGrasp))
        predictedGraspingAxis = branchInterpoationFun(sNext) - branchInterpoationFun(
            sGrasp
        )
        if sNext < sGrasp:
            predictedGraspingAxis = (
                -predictedGraspingAxis
            )  # revert direction if necessary
        predictedGraspingAxis = (
            1 / np.linalg.norm(predictedGraspingAxis) * predictedGraspingAxis
        )

        return predictedGraspingPosition, predictedGraspingAxis

    def calculateGraspingAccuracyError(
        self,
        predictedGraspingPositions,
        predictedGraspingAxes,
        groundTruthGraspingPoses,
    ):
        """calculates the grasping accuracy error

        Args:
            predictedGraspingPositions (list): set of predicted grasping positions
            predictedGraspingAxes (list): set of predicted grasping axes describing the axis the robots' x-axis should be aligned with to ensure successful grasping.
            groundTruthGraspingPoses (list): set of corresponding transformation matrices describing the ground truth
        """
        graspingPositionErrors = []
        graspingAngularErrorsInRad = []
        graspingAngularErrorsInGrad = []
        projectedGraspingAngularErrorsOnXInRad = []
        projectedGraspingAngularErrorsOnXInGrad = []
        projectedGraspingAngularErrorsOnYInRad = []
        projectedGraspingAngularErrorsOnYInGrad = []
        projectedGraspingAngularErrorsOnZInRad = []
        projectedGraspingAngularErrorsOnZInGrad = []
        groundTruthGraspingAxes = []

        for i in range(0, len(groundTruthGraspingPoses)):
            groundTruthGraspingPosition = groundTruthGraspingPoses[i][:3, 3]
            robotGripperAxisX = groundTruthGraspingPoses[i][:3, 0]
            robotGripperAxisY = groundTruthGraspingPoses[i][:3, 1]
            robotGripperAxisZ = groundTruthGraspingPoses[i][:3, 2]
            groundTruthGraspingAxes.append(robotGripperAxisX)

            # positional error
            graspingPositionErrorVector = (
                groundTruthGraspingPosition - predictedGraspingPositions[i]
            )
            graspingPositionError = np.linalg.norm(graspingPositionErrorVector)
            graspingPositionErrors.append(graspingPositionError)

            # angular error between predicted and measured axis
            dotProduct = np.dot(robotGripperAxisX, predictedGraspingAxes[i])
            # aligtn the direction if direction is inverted
            if dotProduct < 0:
                dotProduct = -dotProduct
            graspingAngularErrorInRad = np.arccos(dotProduct)
            graspingAngularErrorInGrad = np.degrees(graspingAngularErrorInRad)
            graspingAngularErrorsInRad.append(graspingAngularErrorInRad)
            graspingAngularErrorsInGrad.append(graspingAngularErrorInGrad)

            # projected angular errors
            rotAxis = np.cross(robotGripperAxisX, predictedGraspingAxes[i])
            rotAxisNorm = 1 / np.linalg.norm(rotAxis) * rotAxis
            rotVec = graspingAngularErrorInGrad * rotAxisNorm
            r = R.from_rotvec(rotVec, degrees=True)
            # project on X
            projectedAngularGraspingErrorX = np.dot(r.as_rotvec(), robotGripperAxisX)
            projectedAngularGraspingErrorXInRad = np.linalg.norm(
                projectedAngularGraspingErrorX
            )
            projectedAngularGraspingErrorXInGrad = np.degrees(
                projectedAngularGraspingErrorXInRad
            )
            projectedGraspingAngularErrorsOnXInRad.append(
                projectedAngularGraspingErrorXInRad
            )
            projectedGraspingAngularErrorsOnXInGrad.append(
                projectedAngularGraspingErrorXInGrad
            )
            # project on Y
            projectedAngularGraspingErrorY = np.dot(r.as_rotvec(), robotGripperAxisY)
            projectedAngularGraspingErrorYInRad = np.linalg.norm(
                projectedAngularGraspingErrorY
            )
            projectedAngularGraspingErrorYInGrad = np.degrees(
                projectedAngularGraspingErrorYInRad
            )
            projectedGraspingAngularErrorsOnYInRad.append(
                projectedAngularGraspingErrorYInRad
            )
            projectedGraspingAngularErrorsOnYInGrad.append(
                projectedAngularGraspingErrorYInGrad
            )
            # project on Z
            projectedAngularGraspingErrorZ = np.dot(r.as_rotvec(), robotGripperAxisZ)
            projectedAngularGraspingErrorZInRad = np.linalg.norm(
                projectedAngularGraspingErrorZ
            )
            projectedAngularGraspingErrorZInGrad = np.degrees(
                projectedAngularGraspingErrorZInRad
            )
            projectedGraspingAngularErrorsOnZInRad.append(
                projectedAngularGraspingErrorZInRad
            )
            projectedGraspingAngularErrorsOnZInGrad.append(
                projectedAngularGraspingErrorZInGrad
            )

        return (
            graspingPositionErrors,
            graspingAngularErrorsInRad,
            graspingAngularErrorsInGrad,
            projectedGraspingAngularErrorsOnXInRad,
            projectedGraspingAngularErrorsOnXInGrad,
            projectedGraspingAngularErrorsOnYInRad,
            projectedGraspingAngularErrorsOnYInGrad,
            projectedGraspingAngularErrorsOnZInRad,
            projectedGraspingAngularErrorsOnZInGrad,
        )
