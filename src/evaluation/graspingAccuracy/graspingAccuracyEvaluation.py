import sys
import os
import numpy as np

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
                (graspingPosition["branch"], graspingPosition["s"])
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
        self, T, graspingLocalCoordinates: tuple
    ):
        """Predict the grasping position and axis from registered target positions

        Args:
            T (np.array): Registered target positions
            B ()
            graspingLocalCoordinates (tuple): local branch coordinates as (b,s) where b is the branch index and s is the local coordiante along the branch.
        """
        correspondingIndices = [
            index
            for index, value in enumerate(initializationResult["localization"]["BInit"])
            if value
            == graspingLocalCoordinate[0]
            - 1  # account for branch indexing starting @ 1 in model desciption
        ]
