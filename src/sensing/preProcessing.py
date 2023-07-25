import os
import sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    from src.sensing.pointCloudProcessing import PointCloudProcessing
    from src.sensing.imageProcessing import ImageProcessing
except:
    print("Imports for class PointCloudPorcessing failed.")
    raise


class PreProcessing(PointCloudProcessing, ImageProcessing):
    """Class providing functions for preprocessing of camera data"""

    def __init__(
        self,
        calibrationParameters=None,
        cameraParameters=None,
        hsvFilterParameters=None,
        roiFilterParameters=None,
        boundingBoxParameters=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.calibrationParameters = (
            self.loadCalibrationParameters(
                fileName="calibrationParameters.json", folderPath="config/calibration/"
            )
            if calibrationParameters is None
            else calibrationParameters
        )

        self.cameraParameters = (
            self.loadCameraParameters("cameraParameters.json")
            if cameraParameters is None
            else cameraParameters
        )
        self.hsvFilterParameters = (
            self.getFilterDefaultValues_HSV()
            if hsvFilterParameters is None
            else hsvFilterParameters
        )
        self.roiFilterParameters = (
            self.getFilterDefaultValues_ROI()
            if roiFilterParameters is None
            else roiFilterParameters
        )
        self.boundingBoxParameters = (
            self.getBoundingBoxDefaultValues()
            if boundingBoxParameters is None
            else boundingBoxParameters
        )

    def getFilterDefaultValues_HSV(self):
        hsvDefaultValues = {
            "hueMin": 0,
            "hueMax": 180,
            "saturationMin": 0,
            "saturationMax": 255,
            "valueMin": 0,
            "valueMax": 255,
        }
        return hsvDefaultValues

    def getFilterDefaultValues_ROI(self):
        roiDefaultValues = {"uMin": 0, "uMax": 1, "vMin": 0, "vMax": 1}
        return roiDefaultValues

    def getBoundingBoxDefaultValues(self):
        boundingBoxParameters = {
            "xMin": -1,
            "xMax": 1,
            "yMin": -1,
            "yMax": 1,
            "zMin": 0,
            "zMax": 2.5,
        }
        return boundingBoxParameters

    def getParametersFromDict_HSV(self, parameterDict):
        """extracts the filter parameters for a HSV filter from a parameter dict"""
        hueMin = parameterDict["hueMin"]
        hueMax = parameterDict["hueMax"]
        saturationMin = parameterDict["saturationMin"]
        saturationMax = parameterDict["saturationMax"]
        valueMin = parameterDict["valueMin"]
        valueMax = parameterDict["valueMax"]
        return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax

    def getParametersFromDict_ROI(self, parameterDict):
        """extracts the filter parameters for a ROI filter from a parameter dict"""
        uMin = parameterDict["uMin"]
        uMax = parameterDict["uMax"]
        vMin = parameterDict["vMin"]
        vMax = parameterDict["vMax"]
        return uMin, uMax, vMin, vMax

    def getParametersFromDict_BoundingBox(self, parameterDict):
        xMin = parameterDict["xMin"]
        xMax = parameterDict["xMax"]
        yMin = parameterDict["yMin"]
        yMax = parameterDict["yMax"]
        zMin = parameterDict["zMin"]
        zMax = parameterDict["zMax"]
        return xMin, xMax, yMin, yMax, zMin, zMax

    def calculatePointCloudFiltered_2D(
        self,
        rgbImage: np.array,
        disparityMap: np.array,
        qmatrix: np.ndarray = None,
        hsvFilterParameters: dict = None,
        roiFilterParameters: dict = None,
    ):
        if hsvFilterParameters is None:
            hsvFilterParameters = self.hsvFilterParameters
        if roiFilterParameters is None:
            roiFilterParametes = self.roiFilterParameters
        if qmatrix is None:
            qmatrix = self.cameraParameters["qmatrix"]
        # Color Filter
        (
            hueMin,
            hueMax,
            saturationMin,
            saturationMax,
            valueMin,
            valueMax,
        ) = self.getParametersFromDict_HSV(hsvFilterParameters)
        maskFilter_Color = self.getMaskFromRGB_applyHSVFilter(
            rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
        )
        # ROI Filter
        (uMin, uMax, vMin, vMax) = self.getParametersFromDict_ROI(roiFilterParametes)
        maskFilter_ROI = self.getMaskFromRGB_applyROI(rgbImage, uMin, uMax, vMin, vMax)
        combinedMask = self.combineMasks_AND([maskFilter_Color, maskFilter_ROI])
        # PointCloud Generation
        points, colors = self.calculatePointCloud(
            rgbImage, disparityMap, qmatrix, combinedMask
        )
        return points, colors

    def calculatePointCloudFiltered_2D_3D(
        self,
        rgbImage: np.array,
        disparityMap: np.array,
        qmatrix: np.ndarray = None,
        hsvFilterParameters: dict = None,
        roiFilterParameters: dict = None,
        boundingBoxParameters: dict = None,
    ):
        if boundingBoxParameters is None:
            boundingBoxParameters = self.boundingBoxParameters
        if qmatrix is None:
            qmatrix = self.cameraParameters["qmatrix"]
        # PointCloud
        points, colors = self.calculatePointCloudFiltered_2D(
            rgbImage, disparityMap, qmatrix, hsvFilterParameters, roiFilterParameters
        )
        # Bounding Box Filter
        (xMin, xMax, yMin, yMax, zMin, zMax) = self.getParametersFromDict_BoundingBox(
            boundingBoxParameters
        )
        mask_BoundingBox = self.getMaskFromBoundingBox(
            points, xMin, xMax, yMin, yMax, zMin, zMax
        )
        points = points[mask_BoundingBox, :]
        colors = colors[mask_BoundingBox, :]
        return points, colors

    def getInliersFromBoundingBox(self, pointCloud: tuple, boundingBoxParameters: dict):
        """returns a point cloud of inliers for a given boudning box

        Args:
            pointCloud (tuple): point cloud data
            boundingBoxParameters (dict): dictionary containing the parameters of the bounding box

        Returns:
            inliers, colors: point cloud data of the inliers and corresponding color information
        """
        (xMin, xMax, yMin, yMax, zMin, zMax) = self.getParametersFromDict_BoundingBox(
            boundingBoxParameters
        )
        inlierMask = self.getMaskFromBoundingBox(
            pointCloud[0], xMin, xMax, yMin, yMax, zMin, zMax
        )
        inliers = pointCloud[0][inlierMask, :]
        colors = pointCloud[1][inlierMask, :]
        return inliers, colors

    def getOutliersFromBoundingBox(
        self, pointCloud: tuple, boundingBoxParameters: dict
    ):
        """returns a point cloud of outliers for a given bounding box

        Args:
            pointCloud (tuple): point cloud data
            boundingBoxParameters (dict): dictionary containing the parameters of the bounding box

        Returns:
            outliers, colors: point cloud data of the outliers and corresponding color information
        """
        (xMin, xMax, yMin, yMax, zMin, zMax) = self.getParametersFromDict_BoundingBox(
            boundingBoxParameters
        )
        inlierMask = self.getMaskFromBoundingBox(
            pointCloud[0], xMin, xMax, yMin, yMax, zMin, zMax
        )
        outlierMask = np.invert(inlierMask)
        outliers = pointCloud[0][outlierMask, :]
        colors = pointCloud[1][outlierMask, :]
        return outliers, colors

    def transformPointsFromCameraToRobotBaseCoordinates(self, points: np.array):
        """Tranforms a given set of points given in the camera coordinate system into the robot base coordinate system

        Args:
            points (np.array): Set of points with coordinates given in camera corodinate system

        Returns:
            points: Set of points with coordinates measured in the robot base coordinate system
        """
        # T_Camera_To_RobotBase: coordinate transfrom of the robot base with respect to the camera coordinate system
        transformationMatrix = np.linalg.inv(
            self.calibrationParameters["T_Camera_To_RobotBase"]
        )
        transformedPoints = self.transformPoints(points, transformationMatrix)
        return transformedPoints

    def transformPointsFromRobotBaseToCameraCoordinates(self, points: np.array):
        """Tranforms a given set of points given in the camera coordinate system into the robot base coordinate system

        Args:
            points (np.array): Set of points with coordinates given in camera corodinate system

        Returns:
            points: Set of points with coordinates measured in the robot base coordinate system
        """
        # T_Camera_To_RobotBase: coordinate transfrom of the robot base with respect to the camera coordinate system
        transformationMatrix = self.calibrationParameters["T_Camera_To_RobotBase"]
        transformedPoints = self.transformPoints(points, transformationMatrix)
        return transformedPoints
