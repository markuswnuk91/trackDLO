import os
import sys
import numpy as np
import cv2
try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for class ImageProcessing failed.")
    raise

class ImageProcessing(DataHandler):
    """Class providing functions for processing image data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # mask generation methods
    def getMaskFromRGB_applyHSVFilter(self, rgbImage: np.array, hMin, hMax, sMin, sMax, vMin, vMax):
        """Returns a mask for a given rgb image, filtering pixels with values out of the given threshold bounds in hsv space

        Args:
            rgbImage (np.array): RGB image
            hMin (int): lower threshold hue
            hMax (int): upper threshold hue
            sMin (int): lower threshold saturation
            sMax (int): upper threshold saturation
            vMin (int): lower threshold value
            vMax (int): upper threshold value

        Returns:
            mask (np.array): mask
                mask values not within threshold bounds are 0, mask values within threshold bounds are 255
        """
        hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsvImage, (hMin, sMin, vMin), (hMax, sMax, vMax))
        return mask
    
    def getMaskFromRGB_applyROI(self, rgbImage: np.array, uMin, uMax, vMin, vMax):
        """Returns a mask for a given rgb image, filtering the pixels with image coordinates out of the given region of interest (ROI)

        Args:
            rgbImage (np.array): RGB image
            uMin (int): lower boarder of ROI along image coordiante u (along width)
            uMax (int): upper boarder of ROI along image coordiante u (along width)
            vMin (int): lower boarder of ROI along image coordiante v (along height)
            vMax (int): upper boarder of ROI along image coordiante v (along height)
        
        Returns:
        mask (np.array): mask
                mask values not within the ROI are 0, mask values within threshold bounds are 255
        """
        imageHeight = rgbImage.shape[0]
        imageWidth = rgbImage.shape[1]
        mask = np.zeros((imageHeight,imageWidth),dtype=np.uint8)
        mask[vMin:vMax,uMin:uMax] = 255
        return mask
    
    # image filtering methods
    def filterRGB_applyHSVFilter(self, rgbImage: np.array, hMin, hMax, sMin, sMax, vMin, vMax):
        mask = self.getMaskFromRGB_applyHSVFilter(rgbImage, hMin, hMax, sMin, sMax, vMin, vMax)
        filteredImage = cv2.bitwise_and(rgbImage, rgbImage, mask=mask)
        return filteredImage
    
    def filterRGB_applyROI(self, rgbImage: np.array, uMin, uMax, vMin, vMax):
        mask = self.getMaskFromRGB_applyROI(uMin, uMax, vMin, vMax)
        filteredImage = cv2.bitwise_and(rgbImage, rgbImage, mask=mask)
        return filteredImage