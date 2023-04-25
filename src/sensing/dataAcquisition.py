import os
import sys
from warnings import warn
import visiontransfer
import cv2
import matplotlib.pyplot as plt
import datetime

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    from src.sensing.cameraInterface import CameraInterface
except:
    print("Imports for SPR failed.")
    raise

class DataAcquisition(CameraInterface):
    """Class providing higher level functions for data acquisition
    """
    def __init__(self, folderPath):
        super().__init__()
        self.folderPath = folderPath

    def saveRGBImage(self,rgb_image,folderPath,fileName):
        cv2.imwrite(folderPath+fileName+'.png', rgb_image)

    def saveNumpyArrayAsBinary(self,numpyArray,folderPath,fileName):
            """Saves the disparity map as a binary numpy array to the specified folder path.
            Args:
                numpyArray (np.array): disparity map as a numpy array
                folderPath (string): path to the folder where the image shoud be saved
                fileName (string): filename of the image

            Raises:
                ValueError: throws if disparity image has the wrong dimension.
            """
            numpyArray.tofile(folderPath + fileName +".bin")

    def saveDisparityMapAsImage(self,disparityMap,folderPath,fileName):
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
        if len(disparityMap.shape)>=3:
            raise ValueError("Obtained 3 dimensions for each pixel. Expected to obtain only one dimension")
        else:
            cv2.imwrite(folderPath+fileName+'.png', img_disparity)

    
    def saveStereoData(self,rgb_image, disparityMap, folderPath , filename_rgbImage, filename_disparityMap, filename_disparityImage):
        self.saveRGBImage(rgb_image,folderPath ,filename_rgbImage)
        self.saveNumpyArrayAsBinary(disparityMap, folderPath, filename_disparityMap)
        self.saveDisparityMapAsImage(disparityMap , folderPath, filename_disparityImage)
        return
    
    def acquireStereoDataSet(self, folderPath = None):
        if folderPath is None:
            folderPath = self.folderPath
        transfer = self.setupAsyncConnection()
        image_set = self.acquireImageSet(transfer)
        stereoDataSet = self.getStereoDataFromImageSet(image_set)
        #generate unique identifier for dataset
        now = datetime.datetime.now()
        date_time_string = now.strftime("%Y%m%d_%H%M%S")
        fileNameRGB = date_time_string + "_image_rgb"
        fileNameDisparityMap = date_time_string + "_map_disparity"
        fileNameDisparityImage = date_time_string + "_image_disparity"
        self.saveStereoData(rgb_image = stereoDataSet[0],disparityMap = stereoDataSet[1], folderPath = folderPath, filename_rgbImage = fileNameRGB, filename_disparityMap = fileNameDisparityMap, filename_disparityImage = fileNameDisparityImage)
        print("Successfully saved data to: ")
        print(folderPath)
        return