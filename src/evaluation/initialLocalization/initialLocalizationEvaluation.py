import sys
import os
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/initialLocalization", ""))
    from src.evaluation.evaluation import Evaluation
except:
    print("Imports for class InitialLocalizationEvaluation failed.")
    raise


class InitialLocalizationEvaluation(Evaluation):
    def __init__(self, configFilePath, *args, **kwargs):
        super().__init__(configFilePath, *args, **kwargs)

    def loadLabelInfo(self, dataSetFolderPath, labelFolderName=None, fileName=None):
        if fileName is None:
            fileName = "labels.json"
        if labelFolderName is None:
            labelFolderName = "labels"

        filePath = dataSetFolderPath + labelFolderName + "/" + fileName
        labelInformation = self.dataHandler.loadFromJson(filePath)
        return labelInformation

    def getFileNameFromLabelEntry(self, labelEntry):
        return labelEntry["file_upload"].split("-")[1]

    def findCorrespondingLabelEntry(self, fileName, labelsDict):
        for labelInfo in labelsDict:  #
            if self.getFileNameFromLabelEntry(labelInfo) == fileName:
                return labelInfo
        return None

    def loadGroundTruthLabelPixelCoordinates(self, dataSetFilePath):
        # gather information
        dataSetFolderPath = self.dataHandler.getDataSetFolderPathFromRelativeFilePath(
            dataSetFilePath
        )
        fileName = self.dataHandler.getFileNameFromRelativeFilePath(dataSetFilePath)
        # load label information
        labelsDict = self.loadLabelInfo(dataSetFolderPath)

        # extract entry corresponding to result
        labelInfo = self.findCorrespondingLabelEntry(fileName, labelsDict)

        # make sure the labels are in correct order
        groundTruthLabels_inPixelCoordiantes = []
        for i, annotationResult in enumerate(labelInfo["annotations"][0]["result"]):
            if (
                int(annotationResult["value"]["keypointlabels"][0].split("_")[-1])
                != i + 1
            ):
                ValueError(
                    "Label order error. Expected label number: {}, instead got: {}".format(
                        i,
                    )
                )
            # extract label pixel coordinates
            xInPixelCoords = int(
                annotationResult["value"]["x"]
                * annotationResult["original_width"]
                / 100
            )
            yInPixelCoords = int(
                annotationResult["value"]["y"]
                * annotationResult["original_height"]
                / 100
            )
            groundTruthLabels_inPixelCoordiantes.append(
                (xInPixelCoords, yInPixelCoords)
            )
        return groundTruthLabels_inPixelCoordiantes
