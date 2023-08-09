import sys
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import cv2

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

    # def drawGraspingPoses(
    #     self,
    #     rgbImage,
    #     dataSetPath,
    #     graspingPositions3D,
    #     graspingAxes3D,
    #     colors,
    #     gripperWidth=0.1,
    #     fingerWidth=0.3,
    # ):
    #     """draws grasp poses in an image

    #     Args:
    #         rgbImage (np.array): RGB-Image
    #         dataSetPath (str): Path to the data set to obtain required information for inverse stereoprojection
    #         graspPositions3D (_type_): _description_
    #         graspAxes3D (_type_): _description_
    #         colors (_type_): _description_
    #         gripperWidth (float, optional): _description_. Defaults to 0.1.
    #         fingerWidth (float, optional): _description_. Defaults to 0.3.
    #     """
    #     # reproject grasp positions
    #     graspingPositions2D = eval.reprojectFrom3DRobotBase(
    #         graspingPositions3D, dataSetPath
    #     )
    #     return rgbImage
    def visualizeGraspingPoses2D(
        self,
        frame,
        dataSetPath,
        graspingPositions3D: np.array,
        graspingAxes3D: np.array,
        colors: list = None,
        gipperWidth3D=0.1,
        fingerWidth2D=0.5,
        centerThickness=10,
        lineThickness=5,
        markerFill=-1,
    ):
        # reproject grasping positions in image
        graspingPositions2D = self.reprojectFrom3DRobotBase(
            graspingPositions3D, dataSetPath
        )
        # reproject grasping axes in image
        graspingAxesStartPoints3D = (
            graspingPositions3D - gipperWidth3D / 2 * graspingAxes3D
        )
        graspingAxesEndPoints3D = (
            graspingPositions3D + gipperWidth3D / 2 * graspingAxes3D
        )
        graspingAxesStartPoints2D = self.reprojectFrom3DRobotBase(
            graspingAxesStartPoints3D, dataSetPath
        )
        graspingAxesEndPoints2D = self.reprojectFrom3DRobotBase(
            graspingAxesEndPoints3D, dataSetPath
        )
        # 2D grasping axes
        graspingAxes2D = graspingAxesEndPoints2D - graspingAxesStartPoints2D

        rgbImage, _ = self.getDataSet(frame, dataSetPath)

        self.drawGraspingPoses2D(
            rgbImage=rgbImage,
            graspingPositions2D=graspingPositions2D,
            graspingAxes2D=graspingAxes2D,
            colors=colors,
        )
        return rgbImage

    def drawGraspingPoses2D(
        self,
        rgbImage,
        graspingPositions2D: np.array,
        graspingAxes2D: np.array,
        colors: list = None,
        fingerWidth2D=0.5,
        centerThickness=10,
        lineThickness=5,
        markerFill=-1,
    ):
        # ensure we have a 2D array
        graspingPositions2D = graspingPositions2D.reshape(-1, 2)
        graspingAxes2D = graspingAxes2D.reshape(-1, 2)

        if len(colors) != len(graspingPositions2D):
            raise ValueError(
                "Expected same length for lists of grasping positions and colors"
            )
        colors = (
            [[1, 0, 0] for _ in range(len(graspingPositions2D))]
            if colors is None
            else colors
        )
        # convert colors to open cv format
        cvColors = []
        for color in colors:
            cvColors.append([value * 255 for value in color])
        # compute orthogonal 2D gripper axis
        gripperAxes2D = (np.array(([0, 1], [-1, 0])) @ graspingAxes2D.T).T
        # compute start and end points for gripper
        gripperStartPoints2D = np.around(
            (graspingPositions2D - 0.5 * gripperAxes2D)
        ).astype(int)
        gripperEndPoints2D = np.around(
            graspingPositions2D + 0.5 * gripperAxes2D
        ).astype(int)
        # compute start and end points for gripper fingers
        gripperEndFingerStartPoints = np.around(
            gripperEndPoints2D - 0.5 * fingerWidth2D * graspingAxes2D
        ).astype(int)
        gripperEndFingerEndPoints = np.around(
            gripperEndPoints2D + 0.5 * fingerWidth2D * graspingAxes2D
        ).astype(int)
        gripperStartFingerStartPoints = np.around(
            gripperStartPoints2D - 0.5 * fingerWidth2D * graspingAxes2D
        ).astype(int)
        gripperStartFingerEndPoints = np.around(
            gripperStartPoints2D + 0.5 * fingerWidth2D * graspingAxes2D
        ).astype(int)

        # draw
        for i, graspingPosition in enumerate(graspingPositions2D):
            # grasping centers
            rgbImage = cv2.circle(
                rgbImage,
                graspingPosition,
                centerThickness,
                cvColors[i],
                markerFill,
            )
            # draw gripper axes
            rgbImage = cv2.line(
                rgbImage,
                (
                    gripperStartPoints2D[i][0],
                    gripperStartPoints2D[i][1],
                ),
                (
                    gripperEndPoints2D[i][0],
                    gripperEndPoints2D[i][1],
                ),
                cvColors[i],
                lineThickness,
            )
            # finger at end
            rgbImage = cv2.line(
                rgbImage,
                (
                    gripperEndFingerStartPoints[i][0],
                    gripperEndFingerStartPoints[i][1],
                ),
                (
                    gripperEndFingerEndPoints[i][0],
                    gripperEndFingerEndPoints[i][1],
                ),
                cvColors[i],
                lineThickness,
            )
            # finger at start
            rgbImage = cv2.line(
                rgbImage,
                (
                    gripperStartFingerStartPoints[i][0],
                    gripperStartFingerStartPoints[i][1],
                ),
                (
                    gripperStartFingerEndPoints[i][0],
                    gripperStartFingerEndPoints[i][1],
                ),
                cvColors[i],
                lineThickness,
            )
        return rgbImage
