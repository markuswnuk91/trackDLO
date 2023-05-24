import os
import sys
import cv2
import matplotlib.pyplot as plt
import datetime
import json
import numpy as np
import glob

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
except:
    print("Imports for Data Hander failed.")
    raise


class DataHandler(object):
    """Class providing functions for data handling"""

    def __init__(
        self,
        defaultLoadFolderPath=None,
        defaultSaveFolderPath=None,
    ):
        """initialization

        Args:
            dataSetFolderPath (string): provide the folder path to the data set folder containing the parameter files
        """
        self.defaultLoadFolderPath = defaultLoadFolderPath
        self.defaultLoadFolderPath_Data = (
            None
            if defaultLoadFolderPath is None
            else self.defaultLoadFolderPath + "data/"
        )
        self.defaultSaveFolderPath = defaultSaveFolderPath

    # utility functions
    def generateIdentifier(self):
        now = datetime.datetime.now()
        date_time_string = now.strftime("%Y%m%d_%H%M%S_%f")
        return date_time_string

    def checkFileExtension(self, file_list, extension):
        for file_path in file_list:
            if file_path.endswith(extension):
                return True
        return False

    def checkIfDisparityDataIsSavedAsTif(self, fileID, folderPath):
        search_pattern = os.path.join(folderPath, f"{fileID}_*")
        matching_files = glob.glob(search_pattern)
        if self.checkFileExtension(matching_files, ".tif"):
            return True
        elif self.checkFileExtension(matching_files, ".npy"):
            return False
        else:
            raise ValueError(
                "Disparity data not saved as tif or npy for file with ID: {}".format(
                    fileID
                )
            )

    def jsonifyDictionary(self, inputDict):
        outputDict = inputDict.copy()
        for key in outputDict:
            if isinstance(outputDict[key], np.ndarray):
                outputDict[key] = outputDict[key].tolist()
            elif isinstance(outputDict[key], np.float32):
                outputDict[key] = float(outputDict[key])
        return outputDict

    # load functions
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

    def loadDisparityDataFromTIF(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        dispartiy_data = cv2.imread(folderPath + fileName, -1)
        return dispartiy_data

    def loadDisparityMapFromTIF(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath_Data = self.defaultLoadFolderPath_Data
            folderPath = self.defaultLoadFolderPath
        disparity_data = self.loadDisparityDataFromTIF(fileName, folderPath_Data)
        cameraParameters = self.loadCameraParameters(
            "cameraParameters.json", folderPath
        )
        disparityMap = disparity_data / cameraParameters["disparityRangeFactor"]
        return disparityMap

    def loadCameraParameters(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath
        with open(folderPath + fileName, "r") as f:
            cameraParameters = json.load(f)
        cameraParameters["qmatrix"] = np.array(cameraParameters["qmatrix"])
        f.close()
        return cameraParameters

    def loadCalibrationParameters(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath
        with open(folderPath + fileName, "r") as f:
            calibrationParameters = json.load(f)
        calibrationParameters["T_Camera_To_RobotBase"] = np.array(
            calibrationParameters["T_Camera_To_RobotBase"]
        )
        f.close()
        return calibrationParameters

    def loadModelParameters(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath
        with open(folderPath + fileName, "r") as f:
            modelParameters = json.load(f)
            modelParameters["topologyModel"] = np.array(
                modelParameters["topologyModel"]
            )
        return modelParameters

    def loadFromJson(self, filePath):
        with open(filePath, "r") as f:
            data = json.load(f)
        f.close()
        return data

    def loadStereoDataSet_FromRGB(self, rgbFileName, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        fileID = "_".join(rgbFileName.split("_")[:-2])

        rgbImage = self.loadNumpyArrayFromPNG(rgbFileName, folderPath)
        hasTif = self.checkIfDisparityDataIsSavedAsTif(fileID, folderPath)

        if hasTif:
            disparityFileName = fileID + "_" + "image_disparity.tif"
            disparityMap = self.loadDisparityMapFromTIF(disparityFileName)
        else:
            disparityFileName = fileID + "_" + "map_disparity.npy"
            disparityMap = self.loadNumpyArrayFromBinary(disparityFileName, folderPath)
        return (rgbImage, disparityMap)

    def saveRGBImage(self, rgb_image, folderPath, fileName):
        cv2.imwrite(folderPath + fileName + ".png", rgb_image)

    # save functions
    def saveNumpyArrayAsBinary(self, numpyArray, folderPath, fileName):
        """Saves the disparity map as a binary numpy array to the specified folder path.
        Args:
            numpyArray (np.array): disparity map as a numpy array
            folderPath (string): path to the folder where the image shoud be saved
            fileName (string): filename of the image

        Raises:
            ValueError: throws if disparity image has the wrong dimension.
        """
        # numpyArray.tofile(folderPath + fileName +".bin")
        np.save(folderPath + fileName + ".npy", numpyArray)

    def saveDisparityMapAsImage(self, disparityMap, folderPath, fileName):
        """Saves the disparity map from the given image set as .png to the specified folder path.
        CAUTION: The saved disparity values should not be used for point cloud reconstruction.
        They contain a scaling factor to map the range of the disparity values to the grayscale image range of 0-255.

        Args:
            image_set (): visiontransfer image set
            folderPath (string): path to the folder where the image shoud be saved
            fileName (string): filename of the image

        Raises:
            ValueError: throws if disparity image has the wrong dimension.
        """
        img_disparity = self.convertDisparityMapToImage(disparityMap)
        if len(disparityMap.shape) >= 3:
            raise ValueError(
                "Obtained 3 dimensions for each pixel. Expected to obtain only one dimension"
            )
        else:
            cv2.imwrite(folderPath + fileName + ".png", img_disparity)

    def saveDisparityImage(self, img_disparity, folderPath, fileName):
        """Saves the disparity map from the given image set as .tif to the specified folder path.
        CAUTION: The saved disparity values should not be used for point cloud reconstruction.
        They contain a scaling factor to map the range of the disparity values to the grayscale image range of 0-255.

        Args:
            image_set (): visiontransfer image set
            folderPath (string): path to the folder where the image shoud be saved
            fileName (string): filename of the image

        Raises:
            ValueError: throws if disparity image has the wrong dimension.
        """
        if len(img_disparity.shape) >= 3:
            raise ValueError(
                "Obtained 3 dimensions for each pixel. Expected to obtain only one dimension"
            )
        else:
            cv2.imwrite(
                folderPath + fileName + ".tif",
                img_disparity,
                params=(cv2.IMWRITE_TIFF_COMPRESSION, 32946),
            )

    def saveStereoData(
        self,
        rgb_image,
        disparityMap,
        disparity_image,
        folderPath,
        filename_rgbImage,
        filename_disparityMap,
        filename_disparityImage,
        saveDisparityMap=False,
    ):
        self.saveRGBImage(rgb_image, folderPath, filename_rgbImage)
        if saveDisparityMap:
            self.saveNumpyArrayAsBinary(disparityMap, folderPath, filename_disparityMap)
        self.saveDisparityImage(disparity_image, folderPath, filename_disparityImage)
        return

    def saveCameraParameters(self, folderPath, fileName="cameraParameters"):
        cameraParameters = self.jsonifyDictionary(self.cameraParameters)
        self.saveDictionaryAsJson(cameraParameters, folderPath, fileName)

    def saveRobotState(self, robotState: dict, folderPath, fileName):
        robotState = self.jsonifyDictionary(robotState)
        self.saveDictionaryAsJson(robotState, folderPath, fileName)

    def saveDictionaryAsJson(self, metaData: dict, folderPath, fileName):
        with open(folderPath + fileName + ".json", "w") as fp:
            json.dump(metaData, fp, indent=4)

    # getter functions
    def getDataSetFileNames(self, dataSetFolderPath=None, type="rgb"):
        if dataSetFolderPath is None:
            dataSetFolderPath = self.defaultLoadFolderPath
        dataFolderPath = dataSetFolderPath + "data/"

        if type == "rgb":
            fileNames = self.getDataSetFileNames_RBG(dataFolderPath)
        elif type == "npy":
            fileNames = self.getDataSetFileNames_NPY(dataFolderPath)
        elif type == "tif":
            fileNames = self.getDataSetFileNames_TIF(dataFolderPath)
        else:
            raise ValueError(
                "File type expected to be rgb, npy, or tif. Other file types are currently not supported."
            )
        return fileNames

    def getDataSetFileNames_RBG(self, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        dataSetFileNames = []
        for fileName in os.listdir(folderPath):
            if fileName.endswith("rgb.png"):
                dataSetFileNames.append(fileName)
        dataSetFileNames.sort()
        return dataSetFileNames

    def getDataSetFileNames_TIF(self, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        dataSetFileNames = []
        for fileName in os.listdir(folderPath):
            if fileName.endswith(".tif"):
                dataSetFileNames.append(fileName)
        dataSetFileNames.sort()
        return dataSetFileNames

    def getDataSetFileNames_NPY(self, folderPath=None):
        if folderPath is None:
            folderPath = self.defaultLoadFolderPath_Data
        dataSetFileNames = []
        for fileName in os.listdir(folderPath):
            if fileName.endswith(".npy"):
                dataSetFileNames.append(fileName)
        dataSetFileNames.sort()
        return dataSetFileNames

    def getDataSetFileName_RBG(self, index, folderPath=None):
        return self.getDataSetFileNames_RBG(folderPath)[index]

    def getDataSetIndexFromFileName(self, fileName, folderPath=None):
        fileNames = self.getDataSetFileNames_RBG(folderPath)
        return fileNames.index(fileName)

    def getDataSetFolderPathFromRelativeFilePath(self, filePath):
        return "/".join(filePath.split("/")[:-2]) + "/"

    def getDataFolderPathFromRelativeFilePath(self, filePath):
        return "/".join(filePath.split("/")[:-1]) + "/"

    def getFileNameFromRelativeFilePath(self, filePath):
        return filePath.split("/")[-1]

    # setter functions
    def setDefaultLoadFolderPathFromFullFilePath(self, filePath):
        self.defaultLoadFolderPath = self.getDataSetFolderPathFromRelativeFilePath(
            filePath
        )
        self.defaultLoadFolderPath_Data = self.getDataFolderPathFromRelativeFilePath(
            filePath
        )
        return

    def saveFigure(
        self,
        fileName,
        fileType,
        folderPath=None,
    ):
        plt.savefig(
            folderPath + fileName + fileType,
            bbox_inches="tight",
        )
