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
            #numpyArray.tofile(folderPath + fileName +".bin")
            np.save(folderPath + fileName+".npy", numpyArray)

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

    def saveDisparityImage(self,img_disparity,folderPath,fileName):
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
        if len(img_disparity.shape)>=3:
            raise ValueError("Obtained 3 dimensions for each pixel. Expected to obtain only one dimension")
        else:
            cv2.imwrite(folderPath+fileName+'.tif', img_disparity)

    def saveStereoData(self,rgb_image, disparityMap, disparity_image, folderPath , filename_rgbImage, filename_disparityMap, filename_disparityImage, saveDisparityMap = False):
        self.saveRGBImage(rgb_image,folderPath ,filename_rgbImage)
        if saveDisparityMap:
            self.saveNumpyArrayAsBinary(disparityMap, folderPath, filename_disparityMap)
        self.saveDisparityImage(disparity_image , folderPath, filename_disparityImage)
        return
    
    def generateIdentifier(self):
        now = datetime.datetime.now()
        date_time_string = now.strftime("%Y%m%d_%H%M%S_%f")
        return date_time_string
    
    def jsonifyDictionary(self, inputDict):
        outputDict = inputDict.copy()
        for key in outputDict:
            if isinstance(outputDict[key],np.ndarray):
                outputDict[key] = outputDict[key].tolist()
            elif isinstance(outputDict[key], np.float32):
                outputDict[key] = float(outputDict[key])
        return outputDict

    def saveCameraParameters(self, folderPath, fileName = "cameraParameters"):
        cameraParameters = self.jsonifyDictionary(self.cameraParameters)
        self.saveDictionaryAsJson(cameraParameters, folderPath, fileName)

    def saveRobotState(self, robotState: dict, folderPath, fileName):
        robotState = self.jsonifyDictionary(robotState)
        self.saveDictionaryAsJson(robotState, folderPath, fileName)
        
    def saveDictionaryAsJson(self, metaData: dict, folderPath, fileName):
        with open(folderPath + fileName + ".json", "w") as fp:
            json.dump(metaData,fp, indent=4)

    def recordStereoDataSet(self, folderPath = None):
        """Method to acquire a single stereo data set (rgb image, disparityMap, disparityImage)

        Args:
            folderPath (sting, optional): 
                path to the folder where the stereo data should be saved. 
                If None the path in self.folderPath is used.
        """
        if folderPath is None:
            folderPath = self.folderPath
        transfer = self.setupAsyncConnection()
        image_set = self.acquireImageSet(transfer)
        stereoDataSet = self.getStereoDataFromImageSet(image_set)

        #generate unique identifier for dataset
        date_time_string = self.generateIdentifier()
        fileNameRGB = date_time_string + "_image_rgb"
        fileNameDisparityMap = date_time_string + "_map_disparity"
        fileNameDisparityImage = date_time_string + "_image_disparity"

        # saving data
        # meta data
        self.saveCameraParameters(folderPath)
        # stereo data
        self.saveStereoData(rgb_image = stereoDataSet[0],disparityMap = stereoDataSet[1], disparity_image = stereoDataSet[2], folderPath = folderPath +"/data/", filename_rgbImage = fileNameRGB, filename_disparityMap = fileNameDisparityMap, filename_disparityImage = fileNameDisparityImage)
        print("Successfully saved data to: ")
        print(folderPath)
        return

    def recordStereoDataSets(self, folderPath = None, method = "manual", fps = 30):
        """Method to acquire several stereo datasets
        Args:
            folderPath (sting, optional): 
                path to the folder where the stereo data should be saved. 
                If None the path in self.folderPath is used.
            method (str, optional): 
                "manual" for manual acquisition, where a button is pressed to acuire an image
                "auto" for automatic acquisition, where images a recorded until 'ESC' is pressed
                Defaults to "manual".
            fps (int, optional): maximum image acquisition rate for mode "auto". Defaults to 30.
        """
        if folderPath is None:
            folderPath = self.folderPath
        transfer = self.setupAsyncConnection()

        if method == "manual":
            print("Press any key to save stereo data.")
            print("Press 'ESC' to end stereo data acquisition ...")
            dataSetCounter = 0
            transfer = self.setupAsyncConnection()
            run = True
            waitTime = int(1000/fps) # waitTime [ms] = 1000ms / fps
            while run:
                try:
                    image_set = self.acquireImageSet(transfer)
                    rgb_image = self.getRGBDataFromImageSet(image_set)
                    cv2.imshow("RGB image", cv2.resize(rgb_image, None, fx=.25, fy=.25))
                    key = cv2.waitKey(waitTime)
                    if key == 27:#if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
                        print("stopped data acquisition.")
                        break
                    elif key != -1:
                        if dataSetCounter==0:
                            self.saveCameraParameters(folderPath)
                        stereoDataSet = self.getStereoDataFromImageSet(image_set)
                        #generate unique identifier for dataset
                        date_time_string = self.generateIdentifier()
                        fileNameRGB = date_time_string + "_image_rgb"
                        fileNameDisparityMap = date_time_string + "_map_disparity"
                        fileNameDisparityImage = date_time_string + "_image_disparity"
                        self.saveStereoData(rgb_image = stereoDataSet[0],disparityMap = stereoDataSet[1], disparityImage = stereoDataSet[2], folderPath = folderPath, filename_rgbImage = fileNameRGB, filename_disparityMap = fileNameDisparityMap, filename_disparityImage = fileNameDisparityImage)
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
            waitTime = int(1000/fps) # waitTime [ms] = 1000ms / fps
            while run:
                try:
                    image_set = self.acquireImageSet(transfer)
                    rgb_image = self.getRGBDataFromImageSet(image_set)
                    cv2.imshow("RGB image", cv2.resize(rgb_image, None, fx=.25, fy=.25))
                    key = cv2.waitKey(waitTime)
                    if key == 27:#if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
                        print("stopped data acquisition.")
                        break
                    else:
                        if dataSetCounter==0:
                            self.saveCameraParameters(folderPath)
                        stereoDataSet = self.getStereoDataFromImageSet(image_set)
                        #generate unique identifier for dataset
                        date_time_string = self.generateIdentifier()
                        fileNameRGB = date_time_string + "_image_rgb"
                        fileNameDisparityMap = date_time_string + "_map_disparity"
                        fileNameDisparityImage = date_time_string + "_image_disparity"
                        self.saveStereoData(rgb_image = stereoDataSet[0],disparityMap = stereoDataSet[1], disparityImage = stereoDataSet[2], folderPath = folderPath, filename_rgbImage = fileNameRGB, filename_disparityMap = fileNameDisparityMap, filename_disparityImage = fileNameDisparityImage)
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