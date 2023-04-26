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
    print("Imports for Data Hander failed.")
    raise

class DataHandler(object):
    """Class providing functions for data handling
    """
    def __init__(self, folderPath):
        self.folderPath = folderPath

    def loadNumpyArrayFromBinary(self, fileName, folderPath = None):
        if folderPath is None:
            folderPath = self.folderPath
        disparityMap = np.load(folderPath + fileName)
        return disparityMap
    
    def loadNumpyArrayFromPNG(self, fileName, folderPath=None, type = "rgb"):
        if folderPath is None:
            folderPath = self.folderPath
        if type == "rgb":
            imageArray = cv2.imread(folderPath+fileName,cv2.IMREAD_COLOR)
            imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB)
        elif type == "grayscale":
            imageArray = cv2.imread(folderPath+fileName,cv2.IMREAD_GRAYSCALE)
        return imageArray
    
    def loadCameraParameters(self, fileName, folderPath=None):
        if folderPath is None:
            folderPath = self.folderPath
        with open(folderPath+fileName, 'r') as f:
            cameraParameters = json.load(f)  
        cameraParameters["qmatrix"] = np.array(cameraParameters["qmatrix"])
        return cameraParameters