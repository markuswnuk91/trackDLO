import sys
import os

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
