import os
import sys
import cv2
import matplotlib.pyplot as plt
import datetime
import json
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
except:
    print("Imports for Data Hander failed.")
    raise


class DataHandler(object):
    """Class providing functions for data handling"""

    def __init__(self, defaultLoadFolderPath):
        """_summary_

        Args:
            dataSetFolderPath (string): provide the folder path to the data set folder containing the parameter files
        """
        self.defaultLoadFolderPath_Parameters = defaultLoadFolderPath
        self.defaultLoadFolderPath_Data = defaultLoadFolderPath + "data/"

    def loadNumpyArrayFromBinary(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        disparityMap = np.load(folderPath + fileName)
        return disparityMap

    def loadNumpyArrayFromPNG(self, fileName, folderPath=None, type="rgb"):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        if type == "rgb":
            imageArray = cv2.imread(folderPath + fileName, cv2.IMREAD_COLOR)
            imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB)
        elif type == "grayscale":
            imageArray = cv2.imread(folderPath + fileName, cv2.IMREAD_GRAYSCALE)
        return imageArray

    def loadCameraParameters(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Parameters
        with open(folderPath + fileName, "r") as f:
            cameraParameters = json.load(f)
        cameraParameters["qmatrix"] = np.array(cameraParameters["qmatrix"])
        f.close()
        return cameraParameters

    def loadCalibrationParameters(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Parameters
        with open(folderPath + fileName, "r") as f:
            calibrationParameters = json.load(f)
        calibrationParameters["T_Camera_To_RobotBase"] = np.array(
            calibrationParameters["T_Camera_To_RobotBase"]
        )
        f.close()
        return calibrationParameters

    def loadFromJson(self, filePath):
        with open(filePath, "r") as f:
            data = json.load(f)
        f.close()
        return data

    def loadStereoDataSet_FromRGB(self, rgbFileName, folderPath=None):
        if folderPath is None:
            folderpath = self.defaultLoadFolderPath_Data
        fileID = rgbFileName.split("_")[0] + "_" + rgbFileName.split("_")[1]
        disparityMapName = fileID + "_" + "map_disparity.npy"
        rgbImage = self.loadNumpyArrayFromPNG(rgbFileName, folderPath)
        disparityMap = self.loadNumpyArrayFromBinary(disparityMapName, folderPath)
        return (rgbImage, disparityMap)
