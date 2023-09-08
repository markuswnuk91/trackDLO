import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/initialLocalization", ""))
    from src.evaluation.evaluation import Evaluation
except:
    print("Imports for class InitialLocalizationEvaluation failed.")
    raise


class InitialLocalizationEvaluation(Evaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                        i + 1,
                        int(
                            annotationResult["value"]["keypointlabels"][0].split("_")[
                                -1
                            ]
                        ),
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
        return np.array(groundTruthLabels_inPixelCoordiantes)

    def visualizeReprojectionError(
        self,
        fileName,
        dataSetPath,
        modelParameters,
        q,
        markerBranchCoordinates,
        groundTruthLabelCoordinates,
        modelColor=[0, 81 / 255, 158 / 255],
        modelThickness=5,
        reprojectionErrorColor=[1, 0, 0],
        reprojectionErrorThickness=5,
        markerThickness=10,
        markerColor=[0, 190 / 255, 1],
        markerFill=-1,
        groundTruthLabelThickness=10,
        groundTruthLabelColor=[255 / 255, 109 / 255, 106 / 255],
        groundTruthLabelFill=-1,
        imageWitdthInInches=5,
        imageHeightInInches=5,
        plotGrayScale=False,
        block=False,
        save=False,
        savePath="data/eval/imgs/",
        format="png",
        dpi=100,
    ):
        """visualizes the 2D reprojection error between estimated marker coordinates of a configuration and annotated ground truth labels

        Args:
            fileName (str): file name
            dataSetPath (str): path to data set
            modelParameters (dict): parameters of the model
            q (np.array): Nx1 array of generalized coordinates for the N degees of freedom
            markerBranchCoordinates (list of tuples): branch local coordinates as tuples of (branchIndex, s)
            groundTruthLabelCoordinates (np.array): Lx2 of pixel coordinates for the L ground truth labels

        """
        # load image
        rgbImg = self.getDataSet(fileName, dataSetPath)[0]  # load image

        # generate model
        model = self.generateModel(modelParameters)
        # get 3D joint coordinates
        jointPositions3D = np.concatenate(
            model.getAdjacentPointPairs(q=q),
            axis=0,
        )
        # get 3D marker coordinates
        markerCoordinates3D = model.computeForwardKinematicsFromBranchLocalCoordinates(
            q=q,
            branchLocalCoordinates=markerBranchCoordinates,
        )
        # reproject markers in 2D pixel coordinates
        markerCoordinates2D = self.reprojectFrom3DRobotBase(
            markerCoordinates3D, dataSetPath
        )
        # reproject joints in 2D pixel coordinates
        jointPositions2D = self.reprojectFrom3DRobotBase(jointPositions3D, dataSetPath)

        # plot image
        # scale colors
        modelColor = tuple([x * 255 for x in modelColor])
        reprojectionErrorColor = tuple([x * 255 for x in reprojectionErrorColor])
        markerColor = tuple([x * 255 for x in markerColor])
        groundTruthLabelColor = tuple([x * 255 for x in groundTruthLabelColor])

        # draw image
        i = 0
        while i <= len(jointPositions2D[:, 0]) - 1:
            cv2.line(
                rgbImg,
                (jointPositions2D[:, 0][i], jointPositions2D[:, 1][i]),
                (jointPositions2D[:, 0][i + 1], jointPositions2D[:, 1][i + 1]),
                modelColor,
                modelThickness,
            )
            i += 2
        for i, markerPosition in enumerate(markerCoordinates2D):
            cv2.line(
                rgbImg,
                (markerPosition[0], markerPosition[1]),
                (
                    groundTruthLabelCoordinates[i][0],
                    groundTruthLabelCoordinates[i][1],
                ),
                reprojectionErrorColor,
                reprojectionErrorThickness,
            )
            cv2.circle(rgbImg, markerPosition, markerThickness, markerColor, markerFill)
            cv2.circle(
                rgbImg,
                groundTruthLabelCoordinates[i],
                groundTruthLabelThickness,
                groundTruthLabelColor,
                groundTruthLabelFill,
            )

        # use matplotlib
        fig = plt.figure(frameon=False)
        fig.set_size_inches(imageWitdthInInches, imageHeightInInches)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        if plotGrayScale:
            ax.imshow(
                cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY), cmap="gray", aspect="auto"
            )
        else:
            ax.imshow(rgbImg, cmap="gray", aspect="auto")
        if save:
            plt.savefig(
                savePath + fileName.split(".")[0] + "_reprojectionError" + "." + format,
                format=format,
                dpi=dpi,
            )
        plt.show(block=block)

    def evaluateReprojectionError(self, initialLocalizationResult):
        evalResult = {}
        # get the file name corresponding to this result
        dataSetFilePath = initialLocalizationResult["filePath"]
        dataSetPath = initialLocalizationResult["dataSetPath"]
        q = initialLocalizationResult["localizationResult"]["q"]
        modelParameters = initialLocalizationResult["modelParameters"]
        # load the corresponding ground trtuh label coordinates
        groundTruthLabelCoordinates_2D = self.loadGroundTruthLabelPixelCoordinates(
            dataSetFilePath
        )
        # get the local branch coordinates
        markerBranchLocalCoordinates = self.getMarkerBranchLocalCoordinates(dataSetPath)
        # get predicted 3D coordinates
        model = self.generateModel(modelParameters)
        modelInfo = self.dataHandler.loadModelParameters("model.json", dataSetPath)
        markerCoordinates_3D = model.computeForwardKinematicsFromBranchLocalCoordinates(
            q=q,
            branchLocalCoordinates=markerBranchLocalCoordinates,
        )
        # reproject in 2D pixel coordinates
        markerCoordinates_2D = self.reprojectFrom3DRobotBase(
            markerCoordinates_3D, dataSetPath
        )
        # evaluate reprojection error
        reprojectionErrors = groundTruthLabelCoordinates_2D - markerCoordinates_2D
        meanReprojectionError = np.mean(np.linalg.norm(reprojectionErrors, axis=1))

        evalResult["filePath"] = dataSetFilePath
        evalResult["dataSetPath"] = dataSetPath
        evalResult["q"] = q
        evalResult["modelParameters"] = modelParameters
        evalResult["groundTruthLabelCoordinates_2D"] = groundTruthLabelCoordinates_2D
        evalResult["markerBranchLocalCoordinates"] = markerBranchLocalCoordinates
        evalResult["markerCoordinates_3D"] = markerCoordinates_3D
        evalResult["markerCoordinates_2D"] = markerCoordinates_2D
        evalResult["reprojectionErrors"] = reprojectionErrors
        evalResult["meanReprojectionError"] = meanReprojectionError
        return evalResult
