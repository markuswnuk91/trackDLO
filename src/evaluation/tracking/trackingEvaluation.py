import sys, os
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/tracking", ""))
    from src.evaluation.evaluation import Evaluation

    # registration algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
except:
    print("Imports for class TrackingEvaluation failed.")
    raise


class TrackingEvaluation(Evaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {
            "spr": {
                "trackingErrors": [],
                "computationTimes": [],
                "geometricErrors": [],
                "T": [],
            }
        }

    # def setupEvaluationCallback(
    #     self, classHandle, result, visualize=True, saveImages=True
    # ):
    #     if isinstance(classHandle) == StructurePreservedRegistration:
    #         fig, ax = setupVisualization(classHandle.Y.shape[1])
    #         raise NotImplementedError
    #         return partial(
    #             visualizationCallbackTracking,
    #             fig,
    #             ax,
    #             classHandle,
    #             savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/bldoReconstruction/test/",
    #         )
    #     else:
    #         raise NotImplementedError

    def runSPR(
        self,
        PointClouds,
        XInit,
        iterationsUntilUpdate,
        sprParameters,
        vis=True,
        saveResults=True,
        saveImages=False,
    ):
        result = {}
        spr = StructurePreservedRegistration(
            **{
                "X": XInit,
                "Y": PointClouds[0],
            }
        )
        callback = self.setupEvaluationCallback(
            spr, result, visualize=True, saveImages=True
        )
        spr.register(callback)
        return result

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
        collectedLabels = []
        for annotationResult in labelInfo["annotations"][0]["result"]:
            labelNumber = int(
                annotationResult["value"]["keypointlabels"][0].split("_")[-1]
            )
            if len(collectedLabels) > 0 and labelNumber < collectedLabels[-1]:
                ValueError(
                    "Label order error. Expected label number greater than {}, instead got: {}".format(
                        collectedLabels[-1],
                        labelNumber,
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
            collectedLabels.append(labelNumber)

        expectedLabelNumbers = list(
            range(
                1,
                len(self.getModelParameters(dataSetFolderPath)["modelInfo"]["labels"])
                + 1,
            )
        )
        missingLabels = list(set(collectedLabels) ^ set(expectedLabelNumbers))

        return np.array(groundTruthLabels_inPixelCoordiantes), missingLabels
