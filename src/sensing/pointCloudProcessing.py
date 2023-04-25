import os
import sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for Data Hander failed.")
    raise

class PointCloudProcessing(DataHandler):
    """Class providing functions for processing of point cloud data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calculatePointCloudFromImageAndDisparity(self, image: np.ndarray, disparityMap: np.ndarray, qMatrix: np.ndarray):
        if len(image.shape) == 3:
            color = True
            invalidMatch = 511.875
        elif len(image.shape) == 2:
            color = False
            invalidMatch = 255.9375
        else:
            raise ValueError
        disparityMap[disparityMap == 0] = invalidMatch

        imageHeight = image.shape[0]
        imageWidth = image.shape[1]

        vRange = np.array(range(0,imageHeight))
        uRange = np.array(range(0,imageWidth))
        uCoordinateMatrix, vCoordinateMatrix = np.meshgrid(uRange, vRange) # matrices where entries are image coordinates for each pixel

        u = uCoordinateMatrix.flatten("F")
        v = vCoordinateMatrix.flatten("F") 
        d = disparityMap.flatten("F")
        if color:
            colors = np.column_stack((image[:,:,0].flatten("F"), image[:,:,1].flatten("F"), image[:,:,2].flatten("F")))
        else:
            colors = image[0].flatten("F")

        # filter invalid points
        d[d == 0] = invalidMatch
        u = u[d!=invalidMatch]
        v = v[d!=invalidMatch]
        if len(colors.shape)>1:
            colors = colors[d!=invalidMatch,:]
        else:
            colors = colors[d!=invalidMatch]
        d = d[d!=invalidMatch]

        #compute point cloud
        w = (qMatrix[3, 2] * d) + qMatrix[3, 3]
        x = (u * qMatrix[0, 0] + qMatrix[0, 3]) / w
        y = (v * qMatrix[1, 1] + qMatrix[1, 3]) / w
        z =  qMatrix[2, 3] / w
        xyz = np.column_stack((x,y,z))

        pointCloud = np.hstack((xyz, colors.astype(float)))
        return pointCloud