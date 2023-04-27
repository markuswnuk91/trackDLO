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
    """Class providing functions for preprocessing of camera data
    """
    def __init__(self, hsvFilterParameters = None, roiFilterParameters = None, boxFilterParameters = None, *args, **kwargs):
        super().__init__(hsvFilterParameters=None,*args, **kwargs)
        self.hsvFilterParameters = self.getHSVFilterDefaultValues if hsvFilterParameters is None else hsvFilterParameters
    
    def getHSVFilterDefaultValues():
        hsvDefaultValues = {"hueMin":0, "hueMax": 180, "saturationMin":0, "saturationMax":255, "valueMin":0, "valueMax":255}
        return hsvDefaultValues
    
    def getFilteredPointCloudFromImage(self,rgbImage: np.array, hsvFilterParameters: dict = None):
        if hsvFilterParameters is None
            hsvFilterParameters = self.hsvFilterParameters
        # Color Filter
        (hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax) = getParametersFilterColor()
        maskFilter_Color = imageProcessor.getMaskFromRGB_applyHSVFilter(rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax)
        fileredImage_Color = imageProcessor.filterRGB_applyHSVFilter(rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax)