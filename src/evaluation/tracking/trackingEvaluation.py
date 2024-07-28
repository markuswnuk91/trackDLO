import sys, os
import numpy as np
import cv2
import time
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/tracking", ""))
    from src.evaluation.evaluation import Evaluation

    # registration algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import KinematicsPreservingRegistration

    # from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart

    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
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

    # ---------------------------------------------------------------------------
    # ERROR METRIC CALCULATION FUNCTIONS
    # ---------------------------------------------------------------------------

    def calculateTrackingErrors(self, trackingResult):
        frames = []
        trackingErrors = []
        for registrationResult in trackingResult["registrations"]:
            frames.append(
                self.getFileIdentifierFromFilePath(registrationResult["filePath"])
            )
            T = registrationResult["T"]
            Y = registrationResult["Y"]
            trackingError = 1 / len(Y) * np.sum(np.min(distance_matrix(T, Y), axis=0))
            trackingErrors.append(trackingError)
        return trackingErrors

    def calculateGeometricErrors(self, trackingResult):
        geometricErrorResult = {"mean": [], "accumulated": [], "lengthError": []}
        accumulatedGeometricErrorPerIteration = []
        meanGeometricErrorPerIteration = []
        model = self.generateModel(trackingResult["modelParameters"])
        B = trackingResult["B"]
        branchIndices = list(set(B))
        numBranches = len(branchIndices)
        totalLength = 0
        for branch in model.getBranches():
            totalLength += branch.getBranchInfo()["length"]
        correspondingNodeIndices = []
        XRef = model.getCartesianBodyCenterPositions()
        for branchIndex in branchIndices:
            nodeIndices = [i for i, x in enumerate(B) if x == branchIndex]
            correspondingNodeIndices.append(nodeIndices)

        registrationResults = trackingResult["registrations"]
        XInit = registrationResults[0]["X"]
        for registrationResult in registrationResults:
            T = registrationResult["T"]
            geometricBranchErrors = []
            desiredNodeDistances = []
            registeredNodeDistances = []
            referenceNodeDistances = []
            referenceBranchLengths = []
            estimatedBranchLengths = []
            for i, branchIndex in enumerate(branchIndices):
                correspondingNodes = correspondingNodeIndices[i]
                correspondingT = T[correspondingNodes, :]
                correspondingXInit = XInit[correspondingNodes, :]
                correspondingXRef = XRef[correspondingNodes, :]
                referenceDifferences = np.diff(correspondingXRef, axis=0)
                currentDifferences = np.diff(correspondingT, axis=0)
                desiredDifferences = np.diff(correspondingXInit, axis=0)
                currentDistances = np.linalg.norm(currentDifferences, axis=1)
                desiredDistances = np.linalg.norm(desiredDifferences, axis=1)
                referenceDistances = np.linalg.norm(referenceDifferences, axis=1)
                for (
                    desiredNodeDistance,
                    currentNodeDistance,
                    referenceNodeDistance,
                ) in zip(desiredDistances, currentDistances, referenceDistances):
                    registeredNodeDistances.append(currentNodeDistance)
                    desiredNodeDistances.append(desiredNodeDistance)
                    referenceNodeDistances.append(referenceNodeDistance)
                currentBranchLength = np.sum(currentDistances)
                desiredBranchLength = np.sum(desiredDistances)
                referenceBranchLength = np.sum(referenceDistances)
                geometricBranchError = np.abs(desiredBranchLength - currentBranchLength)
                geometricBranchErrors.append(geometricBranchError)
                estimatedBranchLengths.append(currentBranchLength)
                referenceBranchLengths.append(referenceBranchLength)
            geometricError = np.sum(
                np.abs(
                    np.array(desiredNodeDistances) - np.array(registeredNodeDistances)
                )
            )
            meanGeometricError = np.mean(
                np.abs(
                    np.array(desiredNodeDistances) - np.array(registeredNodeDistances)
                )
            )
            estimatedTotalLength = np.sum(estimatedBranchLengths)
            refereceTotalLength = np.sum(referenceBranchLengths)
            branchLengthError = np.abs(refereceTotalLength - estimatedTotalLength)
            geometricErrorResult["accumulated"].append(geometricError)
            geometricErrorResult["mean"].append(meanGeometricError)
            geometricErrorResult["lengthError"].append(branchLengthError)
        return geometricErrorResult

    def calculateReprojectionErrors(self, trackingMethodResult):
        reprojectionErrorResult = {}

        dataSetPath = trackingMethodResult["dataSetPath"]
        # get label local cooridnates
        markerLocalCoordinates = self.getMarkerBranchLocalCoordinates(dataSetPath)
        # deteremine for which frames labels exist
        labelInfo = self.loadLabelInfo(dataSetPath)
        labeledFrames = []
        labeledFramesFileNames = []
        groundTruthPixelCoordinatesForFrame = []
        missingLabelsForFrame = []
        for labelEntry in labelInfo:
            fileName = labelEntry["file_upload"].split("-")[-1]
            labeledFramesFileNames.append(fileName)
            filePath = dataSetPath + "data/" + fileName
            labeledFrames.append(self.getFileIndexFromFileName(fileName, dataSetPath))
            (
                groundTruthPixelCoordinates,
                missingLabels,
            ) = self.loadGroundTruthLabelPixelCoordinates(filePath)
            groundTruthPixelCoordinatesForFrame.append(groundTruthPixelCoordinates)
            missingLabelsForFrame.append(missingLabels)

        trackedFrames = trackingMethodResult["frames"]
        framesToEvaluate = list(set(labeledFrames) & set(trackedFrames))
        framesToEvaluate.sort()
        evaluatedFrames = []
        B = trackingMethodResult["B"]
        S = trackingMethodResult["S"]
        meanReprojectionErrorPerFrame = []
        stdReprojectionErrorPerFrame = []
        reprojectionErrorsPerFrame = []
        predictedCoordinates2DPerFrame = []
        groundTruthCoordinates2DPerFrame = []
        correspondingTrackingMethodResults = []
        targetPositions3D = []
        evaluatedMarkers = []
        for frame in framesToEvaluate:
            groundTruthMarkerCoordinates2D = groundTruthPixelCoordinatesForFrame[
                labeledFrames.index(frame)
            ]
            # get tracking result cooresponding to labeld frame
            correspondingTrackingMethodResult = (
                self.findCorrespondingEntryFromKeyValuePair(
                    trackingMethodResult["registrations"], "frame", frame
                )
            )
            correspondingTrackingMethodResults.append(correspondingTrackingMethodResult)
            T = correspondingTrackingMethodResult["T"]
            targetPositions3D.append(T)
            predictedMarkerPositions3D = self.interpolateRegistredTargets(
                T, B, S, markerLocalCoordinates
            )
            # reproject in 2D pixel coordinates
            predictedMarkerCoordinates2D = self.reprojectFrom3DRobotBase(
                predictedMarkerPositions3D, dataSetPath
            )

            missingLabels = missingLabelsForFrame[labeledFrames.index(frame)]
            markersToEvaluate = list(
                set(list(range(1, len(predictedMarkerPositions3D) + 1)))
                - set(missingLabels)
            )
            evaluatedMarkers.append(markersToEvaluate)
            markerCoordinateIndices = np.array(markersToEvaluate) - 1
            reprojectionErrors = np.linalg.norm(
                predictedMarkerCoordinates2D[markerCoordinateIndices, :]
                - groundTruthMarkerCoordinates2D,
                axis=1,
            )
            meanReprojectionErrorPerFrame.append(np.mean(reprojectionErrors))
            stdReprojectionErrorPerFrame.append(np.std(reprojectionErrors))
            predictedCoordinates2DPerFrame.append(predictedMarkerCoordinates2D)
            groundTruthCoordinates2DPerFrame.append(groundTruthMarkerCoordinates2D)
            reprojectionErrorsPerFrame.append(reprojectionErrors)
            evaluatedFrames.append(frame)
            reprojectionErrorResult["labeledFrames"] = framesToEvaluate
            reprojectionErrorResult["means"] = np.array(meanReprojectionErrorPerFrame)
            reprojectionErrorResult["stds"] = np.array(stdReprojectionErrorPerFrame)
            reprojectionErrorResult["predictedMarkerCoordinates2D"] = (
                predictedCoordinates2DPerFrame
            )
            reprojectionErrorResult["groundTruthMarkerCoordinates2D"] = (
                groundTruthCoordinates2DPerFrame
            )
            reprojectionErrorResult["reprojectionErrors"] = reprojectionErrorsPerFrame
            reprojectionErrorResult["targetPositions3D"] = targetPositions3D
            reprojectionErrorResult["B"] = B
            reprojectionErrorResult["S"] = S
            reprojectionErrorResult["evaluatedMarkers"] = evaluatedMarkers
        return reprojectionErrorResult

    def calculateRuntimes(trackingMethodResult):
        runtimeResults = trackingMethodResult["runtimes"]
        runtimeResults["mean"] = np.mean(
            trackingMethodResult["runtimes"]["runtimesPerIteration"]
        )
        runtimeResults["std"] = np.std(
            trackingMethodResult["runtimes"]["runtimesPerIteration"]
        )
        numPointsPerIterations_Y = []
        numPointsPerIterations_X = []
        numCorrespondancesPerIteration = []
        for registrationResult in trackingMethodResult["registrations"]:
            numPoints_Y = len(registrationResult["Y"])
            numPointsPerIterations_Y.append(numPoints_Y)
            numPoints_X = len(registrationResult["X"])
            numPointsPerIterations_X.append(numPoints_X)
            numCorrespondances = numPoints_Y * numPoints_X
            numCorrespondancesPerIteration.append(numCorrespondances)
        runtimeResults["numPointsPerIteration"] = numPointsPerIterations_Y
        runtimeResults["numCorrespondancesPerIteration"] = (
            numCorrespondancesPerIteration
        )
        return runtimeResults

    # ---------------------------------------------------------------------------
    # VISUALIZATION FUNCITONS
    # ---------------------------------------------------------------------------
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
        pause=3,
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
        if block is False:
            time.sleep(pause)
            plt.close()
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

        rgbImg = plotGraph2_CV(
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
        ax,
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
        elevation=25,
        azimuth=70,
        lineWidth=1.5,
    ):
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
                        linewidth=lineWidth,
                    )
        ax.view_init(elev=elevation, azim=azimuth)
        return ax

    def plotBranchWiseColoredTrackingResult2D(
        self,
        result,
        frame,
        method,
        colorPalette=None,
        lineThickness=None,
        circleRadius=None,
    ):
        colorPalette = (
            thesisColorPalettes["viridis"] if colorPalette is None else colorPalette
        )
        lineThickness = 5 if lineThickness is None else lineThickness
        circleRadius = 10 if circleRadius is None else circleRadius

        # gather parameters
        frame = frame
        dataSetPath = result["dataSetPath"]
        modelParameters = result["trackingResults"][method]["modelParameters"]
        model = self.generateModel(modelParameters)
        rgbImg = self.getDataSet(frame, dataSetPath)[0]

        adjacencyMatrix = result["trackingResults"]["spr"]["adjacencyMatrix"]
        _, _, branchCorrespondanceMatrix = (
            model.getBranchCorrespondancesForSegmentCenters()
        )
        positions3D = result["trackingResults"][method]["registrations"][frame]["T"]
        positions2D = self.reprojectFrom3DRobotBase(positions3D, dataSetPath)
        numBranches = branchCorrespondanceMatrix.shape[1]

        colorScaleCoordinates = np.linspace(0, 1, numBranches)
        branchColors = []
        for s in colorScaleCoordinates:
            branchColors.append(colorPalette.to_rgba(s)[:3])

        for branchIndex in range(0, numBranches):
            indices = np.where(branchCorrespondanceMatrix[:, branchIndex] == 1)[0]
            branchPositions = positions2D[indices, :]
            branchAdjacencyMatrix = np.array(
                [[adjacencyMatrix[row][col] for col in indices] for row in indices]
            )
            rgbImg = plotGraph2_CV(
                rgbImg=rgbImg,
                positions2D=branchPositions,
                adjacencyMatrix=branchAdjacencyMatrix,
                lineColor=branchColors[branchIndex],
                circleColor=branchColors[branchIndex],
                lineThickness=lineThickness,
                circleRadius=circleRadius,
            )
        return rgbImg
