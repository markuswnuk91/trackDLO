import os
import sys
import cv2
import matplotlib.pyplot as plt
import datetime
import json
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    from src.sensing.cameraInterface import CameraInterface
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for SPR failed.")
    raise


class DataAcquisition(DataHandler, CameraInterface):
    """Class providing higher level functions for data acquisition"""

    def __init__(self, defaultSaveFolderPath):
        super().__init__(**{"defaultSaveFolderPath":defaultSaveFolderPath})
        super(DataHandler,self).__init__()

    def recordStereoDataSet(self, folderPath=None, saveDisparityMap=False):
        """Method to acquire a single stereo data set (rgb image, disparityMap, disparityImage)

        Args:
            folderPath (sting, optional):
                path to the folder where the stereo data should be saved.
                If None the path in self.defaultSaveFolderPath is used.
        """
        if folderPath is None:
            folderPath = self.defaultSaveFolderPath
        transfer = self.setupAsyncConnection()
        image_set = self.acquireImageSet(transfer)
        stereoDataSet = self.getStereoDataFromImageSet(image_set)

        # generate unique identifier for dataset
        date_time_string = self.generateIdentifier()
        fileNameRGB = date_time_string + "_image_rgb"
        fileNameDisparityMap = date_time_string + "_map_disparity"
        fileNameDisparityImage = date_time_string + "_image_disparity"

        # saving data
        # meta data
        self.saveCameraParameters(folderPath)
        # stereo data
        self.saveStereoData(
            rgb_image=stereoDataSet[0],
            disparityMap=stereoDataSet[1],
            disparity_image=stereoDataSet[2],
            folderPath=folderPath + "/data/",
            filename_rgbImage=fileNameRGB,
            filename_disparityMap=fileNameDisparityMap,
            filename_disparityImage=fileNameDisparityImage,
            saveDisparityMap=saveDisparityMap,
        )
        print("Successfully saved data to: ")
        print(folderPath)
        return

    def recordStereoDataSets(self, folderPath=None, method="manual", fps=30):
        """Method to acquire several stereo datasets
        Args:
            folderPath (sting, optional):
                path to the folder where the stereo data should be saved.
                If None the path in self.defaultSaveFolderPath is used.
            method (str, optional):
                "manual" for manual acquisition, where a button is pressed to acuire an image
                "auto" for automatic acquisition, where images a recorded until 'ESC' is pressed
                Defaults to "manual".
            fps (int, optional): maximum image acquisition rate for mode "auto". Defaults to 30.
        """
        if folderPath is None:
            folderPath = self.defaultSaveFolderPath
        transfer = self.setupAsyncConnection()

        if method == "manual":
            print("Press any key to save stereo data.")
            print("Press 'ESC' to end stereo data acquisition ...")
            dataSetCounter = 0
            transfer = self.setupAsyncConnection()
            run = True
            waitTime = int(1000 / fps)  # waitTime [ms] = 1000ms / fps
            while run:
                try:
                    image_set = self.acquireImageSet(transfer)
                    rgb_image = self.getRGBDataFromImageSet(image_set)
                    cv2.imshow(
                        "RGB image", cv2.resize(rgb_image, None, fx=0.25, fy=0.25)
                    )
                    key = cv2.waitKey(waitTime)
                    if key == 27:  # if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
                        print("stopped data acquisition.")
                        break
                    elif key != -1:
                        if dataSetCounter == 0:
                            self.saveCameraParameters(folderPath)
                        stereoDataSet = self.getStereoDataFromImageSet(image_set)
                        # generate unique identifier for dataset
                        date_time_string = self.generateIdentifier()
                        fileNameRGB = date_time_string + "_image_rgb"
                        fileNameDisparityMap = date_time_string + "_map_disparity"
                        fileNameDisparityImage = date_time_string + "_image_disparity"
                        self.saveStereoData(
                            rgb_image=stereoDataSet[0],
                            disparityMap=stereoDataSet[1],
                            disparityImage=stereoDataSet[2],
                            folderPath=folderPath,
                            filename_rgbImage=fileNameRGB,
                            filename_disparityMap=fileNameDisparityMap,
                            filename_disparityImage=fileNameDisparityImage,
                        )
                        dataSetCounter += 1
                        print("Acquired data sets: {}".format(dataSetCounter))
                except:
                    cv2.destroyAllWindows()
                    print("Stopped data acquisition due to exception")
                    run = False
                    break
        elif method == "auto":
            print("Press 'ESC' to end stereo data acquisition")
            dataSetCounter = 0
            transfer = self.setupAsyncConnection()
            run = True
            waitTime = int(1000 / fps)  # waitTime [ms] = 1000ms / fps
            while run:
                try:
                    image_set = self.acquireImageSet(transfer)
                    rgb_image = self.getRGBDataFromImageSet(image_set)
                    cv2.imshow(
                        "RGB image", cv2.resize(rgb_image, None, fx=0.25, fy=0.25)
                    )
                    key = cv2.waitKey(waitTime)
                    if key == 27:  # if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
                        print("stopped data acquisition.")
                        break
                    else:
                        if dataSetCounter == 0:
                            self.saveCameraParameters(folderPath)
                        stereoDataSet = self.getStereoDataFromImageSet(image_set)
                        # generate unique identifier for dataset
                        date_time_string = self.generateIdentifier()
                        fileNameRGB = date_time_string + "_image_rgb"
                        fileNameDisparityMap = date_time_string + "_map_disparity"
                        fileNameDisparityImage = date_time_string + "_image_disparity"
                        self.saveStereoData(
                            rgb_image=stereoDataSet[0],
                            disparityMap=stereoDataSet[1],
                            disparityImage=stereoDataSet[2],
                            folderPath=folderPath,
                            filename_rgbImage=fileNameRGB,
                            filename_disparityMap=fileNameDisparityMap,
                            filename_disparityImage=fileNameDisparityImage,
                        )
                        dataSetCounter += 1
                        print("Acquired data sets: {}".format(dataSetCounter))
                except:
                    cv2.destroyAllWindows()
                    print("Stopped data acquisition due to exception")
                    run = False
                    break
        else:
            raise NotImplementedError
        return
