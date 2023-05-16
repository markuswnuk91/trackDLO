import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from warnings import warn
try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    import visiontransfer
    import cv2
except:
    print("Imports for SPR failed.")
    raise

class CameraInterface (object):
    """Class used to interface the Nerian Scarlet stereo camera system
    """
    def __init__(self):
        """Initialization: Finding the camera and retrieving the camera parameters
        """
        device_enum = visiontransfer.DeviceEnumeration()
        devices = device_enum.discover_devices()
        if len(devices) < 1:
            print('No devices found')
            sys.exit(1)

        print('Found these devices:')
        for i, info in enumerate(devices):
            print(f'  {i+1}: {info}')
        selected_device = 0 if len(devices)==1 else (int(input('Device to open: ') or '1')-1)
        print(f'Selected device #{selected_device+1}')
        self.device = devices[selected_device]
        print('Ask parameter server to set stereo mode ...')
        self.params = visiontransfer.DeviceParameters(self.device)
        self.params.set_operation_mode(visiontransfer.OperationMode.STEREO_MATCHING)
        self.cameraParameters = self.getCameraParameters()
        print('Got the follwing camera parameters: {}'.format(self.cameraParameters["qmatrix"]))
        for key in self.cameraParameters:
            print('{}: {}'.format(key, self.cameraParameters[key]))

    def getCameraParameters(self):
        cameraParameters = {}
        print('Acquiring test image to determine camera parameters ...')
        transfer = visiontransfer.AsyncTransfer(self.device)
        image_set = None
        while image_set is None:
            try:
                image_set = transfer.collect_received_image_set()
            except:
                warn("Failed to capture image. Trying again.")
        #get camera parameters
        #height
        height = image_set.get_height()
        # width
        width = image_set.get_width()
        # Q-matrix
        Q = image_set.get_qmatrix()
        # disparityRangeFactor to retrieve the disparity values from neriam image sets
        disprange = image_set.get_disparity_range()[1]
        if disprange == 511:
            disparityRangeFactor = 8  
        elif disprange == 255:
            disparityRangeFactor = 16
        else:
            raise ValueError
        # disparity2ImgScalingFactor to convert disparity value in 0-255 image range
        pixeldata_disparity = (
            image_set.get_pixel_data(
                visiontransfer.ImageType.IMAGE_DISPARITY, do_copy=True
            )
            / disparityRangeFactor
        )
        disparity2ImgScalingFactor = 1/np.max(pixeldata_disparity)*255 

        # aggregate parameters
        cameraParameters["width"] = height
        cameraParameters["height"] = width
        cameraParameters["qmatrix"] = Q
        cameraParameters["cx"] = -Q[0, 3] # According to https://stackoverflow.com/questions/27374970/q-matrix-for-the-reprojectimageto3d-function-in-opencv
        cameraParameters["cy"] = -Q[1, 3]
        cameraParameters["fx"] = Q[2, 3]
        cameraParameters["fy"] = Q[2, 3]
        cameraParameters["Tx"] = -1/Q[3, 2]
        cameraParameters["disparityRangeFactor"] = disparityRangeFactor
        cameraParameters["disparity2ImgScalingFactor"] = disparity2ImgScalingFactor
        print('Successfully extracted camera parameters ...')
        return cameraParameters
    
    def setupAsyncConnection(self):
        transfer = visiontransfer.AsyncTransfer(self.device)
        return transfer
    
    def acquireImageSet(self,transfer):
        image_set = transfer.collect_received_image_set()
        self.image_set = image_set # make this image set to the current image set of this class
        return image_set
    
    def acquireImageSets(self, numImages = 1, callback=None):
        """Acquires a specified number of stereo images
        Args:
            numImages (int, optional): number of images which should be acquired. Defaults to 1. Use -1 to loop indefinetely.
            callback (function): callback function which is executed each time a image set is received

        Raises:
            ValueError: _description_
            NotImplementedError: _description_

        Return:
            image_set(visiontransfer image set): 
                acquired image set with stereo image data
        """
        print('Starting acquisition ...')
        transfer = self.setupAsyncTransfer()
        i = 0
        while True:
            if i>=numImages and i != -1: break
            #retrive image set (left and right image)
            image_set = self.acquireImageSet(self,transfer)
            if callable(callback):
                callback()
        return image_set
    
    def getRGBDataFromImageSet(self,image_set):
        # get image data from left camera
        pixeldata_left = image_set.get_pixel_data(
            visiontransfer.ImageType.IMAGE_LEFT, force8bit=True, do_copy=True
        )
        if len(pixeldata_left.shape)==3:
            # Color image: Nerian API uses RGB, OpenCV uses BGR
            img_left_rbg = cv2.cvtColor(pixeldata_left, cv2.COLOR_RGB2BGR)
        else:
            img_left_rbg = np.stack(pixeldata_left,pixeldata_left,pixeldata_left,axis=2)
        return img_left_rbg
    
    def getDisparityDataFromImageSet(self, image_set):
        pixeldata_disparity = (
            image_set.get_pixel_data(
                visiontransfer.ImageType.IMAGE_DISPARITY, do_copy=True
            )
        )
        return pixeldata_disparity
      
    def getDisparityMapFromImageSet(self,image_set):
        pixeldata_disparity = (
                image_set.get_pixel_data(
                    visiontransfer.ImageType.IMAGE_DISPARITY, do_copy=True
                )
                / self.cameraParameters["disparityRangeFactor"]
            )
        return pixeldata_disparity
    
    def convertDisparityMapToImage(self,dispMap):
        return (dispMap.copy() * self.cameraParameters["disparity2ImgScalingFactor"]).astype(np.uint8)
    
    def getStereoDataFromImageSet(self, image_set):
        """extracts the relevant stereo image data  from a given image set.

        Args:
            image_set (visiontransfer image set): image set

        Returns:
            rgbImg (np.array): 
                RGB-image as array of 0-255 RGB values
            dispMap (np.array):
                dispartiy map as array of disparity values (required to calculate 3D point cloud data)
            dispImg (np.array):
                disparity image as array of 0-255 RGB (not usable to caluclate 3D point cloud data)
        """
        rgbImg = self.getRGBDataFromImageSet(image_set)
        dispMap = self.getDisparityMapFromImageSet(image_set)
        dispImg = self.getDisparityDataFromImageSet(image_set)
        return rgbImg, dispMap, dispImg
    
    def displayLeftRBGImage(self):
        transfer = self.setupAsyncConnection()
        image_set = self.acquireImageSet(transfer)
        rgb_image = self.getRGBDataFromImageSet(image_set)
        cv2.imshow("RGB image", cv2.resize(rgb_image, None, fx=.25, fy=.25))
        cv2.waitKey(0)
        return
    
    def streamLeftRBGImage(self, fps = 30):
        transfer = self.setupAsyncConnection()
        run = True
        print("Press 'ESC' to stop image stream ...")
        waitTime = int(1000/fps) # waitTime [ms] = 1000ms / fps
        while run:
            try:
                key = cv2.waitKey(waitTime)
                if key == 27:#if ESC is pressed, exit loop
                    print("Stopped streaming.")
                    break
                image_set = self.acquireImageSet(transfer)
                rgb_image = self.getRGBDataFromImageSet(image_set)
                cv2.imshow("RGB image", cv2.resize(rgb_image, None, fx=.25, fy=.25))
            except:
                print("Stopped streaming due to exception")
                run = False
                break
        return
    
    def saveRGBImageFromImageSet(self,image_set,folderPath,fileName):
        img_left_rbg = self.getRGBDataFromImageSet(image_set)
        cv2.imwrite(folderPath+fileName+'.png', img_left_rbg)

    def saveDisparityMapFromImageSet(self,image_set,folderPath,fileName):
            """Saves the disparity map from the given image set as a binary numpy array to the specified folder path.
            Use the disparity values from this numpy array to reconstruct 3D point cloud data.
            Args:
                image_set (): visiontransfer image set
                folderPath (string): path to the folder where the image shoud be saved
                fileName (string): filename of the image

            Raises:
                ValueError: throws if disparity image has the wrong dimension.
            """
            pixeldata_disparity = (
                image_set.get_pixel_data(
                    visiontransfer.ImageType.IMAGE_DISPARITY, do_copy=True
                )
                / self.cameraParameters["disparityRangeFactor"]
            )
            pixeldata_disparity.tofile(folderPath + fileName +".bin")

    def saveDisparityImageFromImageSet(self,image_set,folderPath,fileName):
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
        img_disparity = self.getDisparityDataFromImageSet(image_set)
        if len(img_disparity.shape)>1:
            raise ValueError("Obtained 3 dimensions for each pixel. Expected to obtain only one dimension")
        else:
            cv2.imwrite(folderPath+fileName+'.tif', img_disparity)

    def plotRBGImageFromImageSet(self, image_set, waitTime = None):
        """Using matplot lib to plot the RGB image of the given image set.

        Args:
            image_set: the acquired image set
            waitTime (int, optional): Wait time in s. Defaults to None.
        """
        rgb_image = self.getRGBDataFromImageSet(image_set)
        # Color image: Nerian API uses RGB, OpenCV uses BGR
        img_left_rbg = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        plt.imshow(img_left_rbg)
        if waitTime == -1:
            plt.show(block=True)
        elif waitTime is None:
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.show(block=False)
            plt.pause(waitTime)