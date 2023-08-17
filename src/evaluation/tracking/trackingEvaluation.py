import sys, os
import numpy as np
import cv2

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/tracking", ""))
    from src.evaluation.evaluation import Evaluation

    # registration algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart

    from src.visualization.plot3D import *
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

    # def runSPR(
    #     self,
    #     PointClouds,
    #     XInit,
    #     iterationsUntilUpdate,
    #     sprParameters,
    #     vis=True,
    #     saveResults=True,
    #     saveImages=False,
    # ):
    #     result = {}
    #     spr = StructurePreservedRegistration(
    #         **{
    #             "X": XInit,
    #             "Y": PointClouds[0],
    #         }
    #     )
    #     callback = self.setupEvaluationCallback(
    #         spr, result, visualize=True, saveImages=True
    #     )
    #     spr.register(callback)
    #     return result

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

    def visualizeReprojectionError(
        self,
        fileName,
        dataSetPath,
        positions3D,
        adjacencyMatrix,
        predictedMarkerCoordinates2D,
        groundTruthMarkerCoordinates2D,
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
        # scale colors
        modelColor = tuple([x * 255 for x in modelColor])
        reprojectionErrorColor = tuple([x * 255 for x in reprojectionErrorColor])
        markerColor = tuple([x * 255 for x in markerColor])
        groundTruthLabelColor = tuple([x * 255 for x in groundTruthLabelColor])

        # reproject joints in 2D pixel coordinates
        positions2D = self.reprojectFrom3DRobotBase(positions3D, dataSetPath)

        # load image
        rgbImg = self.getDataSet(fileName, dataSetPath)[0]  # load image

        # draw image
        i = 0
        j = 0
        I, J = adjacencyMatrix.shape
        for i in range(0, I):
            for j in range(0, J):
                if adjacencyMatrix[i, j] == 1:
                    cv2.line(
                        rgbImg,
                        (
                            positions2D[i, 0],
                            positions2D[i, 1],
                        ),
                        (
                            positions2D[j, 0],
                            positions2D[j, 1],
                        ),
                        modelColor,
                        modelThickness,
                    )
        for i, _ in enumerate(groundTruthMarkerCoordinates2D):
            cv2.line(
                rgbImg,
                (
                    predictedMarkerCoordinates2D[i][0],
                    predictedMarkerCoordinates2D[i][1],
                ),
                (
                    groundTruthMarkerCoordinates2D[i][0],
                    groundTruthMarkerCoordinates2D[i][1],
                ),
                reprojectionErrorColor,
                reprojectionErrorThickness,
            )
            cv2.circle(
                rgbImg,
                predictedMarkerCoordinates2D[i],
                markerThickness,
                markerColor,
                markerFill,
            )
            cv2.circle(
                rgbImg,
                groundTruthMarkerCoordinates2D[i],
                groundTruthLabelThickness,
                groundTruthLabelColor,
                groundTruthLabelFill,
            )

        self.plotImageWithMatplotlib(rgbImg, block=block)
        # while i <= len(registrationTargetCoordinates2D[:, 0]) - 1:
        #     cv2.line(
        #         rgbImg,
        #         (
        #             registrationTargetCoordinates2D[:, 0][i],
        #             registrationTargetCoordinates2D[:, 1][i],
        #         ),
        #         (
        #             registrationTargetCoordinates2D[:, 0][i + 1],
        #             registrationTargetCoordinates2D[:, 1][i + 1],
        #         ),
        #         modelColor,
        #         modelThickness,
        #     )
        #     i += 2
        # for i, markerPosition in enumerate(markerCoordinates2D):
        #     cv2.line(
        #         rgbImg,
        #         (markerPosition[0], markerPosition[1]),
        #         (
        #             groundTruthLabelCoordinates[i][0],
        #             groundTruthLabelCoordinates[i][1],
        #         ),
        #         reprojectionErrorColor,
        #         reprojectionErrorThickness,
        #     )
        #     cv2.circle(rgbImg, markerPosition, markerThickness, markerColor, markerFill)
        #     cv2.circle(
        #         rgbImg,
        #         groundTruthLabelCoordinates[i],
        #         groundTruthLabelThickness,
        #         groundTruthLabelColor,
        #         groundTruthLabelFill,
        #     )

    def drawConfiguration2D(
        self,
        rgbImg,
        positions2D,
        adjacencyMatrix,
        lineColor=[0, 81 / 255, 158 / 255],
        circleColor=[0, 81 / 255, 158 / 255],
        lineThickness=5,
        circleRadius=10,
    ):
        # scale colors
        lineColor = tuple([x * 255 for x in lineColor])
        circleColor = tuple([x * 255 for x in circleColor])

        # draw image
        i = 0
        j = 0
        I, J = adjacencyMatrix.shape
        for i in range(0, I):
            cv2.circle(
                rgbImg, positions2D[i, :], circleRadius, circleColor, thickness=-1
            )
            for j in range(0, J):
                if adjacencyMatrix[i, j] == 1:
                    cv2.line(
                        rgbImg,
                        (
                            positions2D[i, 0],
                            positions2D[i, 1],
                        ),
                        (
                            positions2D[j, 0],
                            positions2D[j, 1],
                        ),
                        lineColor,
                        lineThickness,
                    )
        return rgbImg

    def plotTrackingResult2D(
        self,
        frame,
        dataSetPath,
        positions3D,
        adjacencyMatrix,
        lineColor=[0, 81 / 255, 158 / 255],
        circleColor=[0, 81 / 255, 158 / 255],
        lineThickness=5,
        circleRadius=10,
    ):
        # reproject joints in 2D pixel coordinates
        positions2D = self.reprojectFrom3DRobotBase(positions3D, dataSetPath)

        # load image
        rgbImg = self.getDataSet(frame, dataSetPath)[0]  # load image

        rgbImg = self.drawConfiguration2D(
            rgbImg=rgbImg,
            positions2D=positions2D,
            adjacencyMatrix=adjacencyMatrix,
            lineColor=lineColor,
            circleColor=circleColor,
            lineThickness=lineThickness,
            circleRadius=circleRadius,
        )
        return rgbImg

    def plotTrackingResult3D(
        self,
        pointCloud,
        targets,
        adjacencyMatrix,
        pointCloudColor=[1, 0, 0],
        targetColor=[0, 0, 1],
        lineColor=[0, 0, 1],
        pointCloudPointSize=1,
        targetPointSize=10,
        pointCloudAlpha=0.1,
        targetAlpha=1,
        axisLimX=[0, 1],
        axisLimY=[-0.2, 0.8],
        axisLimZ=[0, 1],
        elevation=25,
        azimuth=70,
    ):
        fig, ax = setupLatexPlot3D(
            axisLimX=axisLimX, axisLimY=axisLimY, axisLimZ=axisLimZ
        )

        plotPointSet(
            ax=ax,
            X=pointCloud,
            color=pointCloudColor,
            size=pointCloudPointSize,
            alpha=pointCloudAlpha,
        )
        plotPointSet(
            ax=ax, X=targets, color=targetColor, size=targetPointSize, alpha=targetAlpha
        )
        i = 0
        j = 0
        I, J = adjacencyMatrix.shape
        for i in range(0, I):
            for j in range(0, J):
                if adjacencyMatrix[i, j] == 1:
                    plotLine(
                        ax=ax,
                        pointPair=np.vstack((targets[i, :], targets[j, :])),
                        color=lineColor,
                    )
        ax.view_init(elev=elevation, azim=azimuth)
        return fig, ax
