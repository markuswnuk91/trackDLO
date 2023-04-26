import os
import sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for class PointCloudPorcessing failed.")
    raise

class PointCloudProcessing(DataHandler):
    """Class providing functions for processing of point cloud data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calculatePointCloud(self, image: np.ndarray, disparityMap: np.ndarray, qMatrix: np.ndarray, mask = None):
        """calculates a point cloud from a given image and disparity map

        Args:
            image (np.ndarray): WidthxHeightxDim image
            disparityMap (np.ndarray): WidthxHeightx1 dispartiy map
            qMatrix (np.ndarray): 4x4 Q matrix for stereo projection
            mask (np.ndarray): WidthxHeightx1 mask, 0 for pixels which should not be projected
        Returns:
            pointCloud(tuple): point cloud information
                pointCloud[0] Mx3 np.array with 3D xyz-coordinates of the points,
                pointCloud[1] Mx3 or Mx1 np.array of color information, depending if input image is color or grayscale
        """
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

        # apply mask
        if mask is not None:
            maskedPixels = mask.flatten("F")
            d = d[maskedPixels!=0]
            u = u[maskedPixels!=0]
            v = v[maskedPixels!=0]
            if len(colors.shape)>1:
                colors = colors[maskedPixels!=0,:]
            else:
                colors = colors[maskedPixels!=0]

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
        xyz = self.stereoProjection(u,v,d,qMatrix)

        return xyz, colors
    
    def stereoProjection(self, uVector,vVector,dVector,qMatrix):
        """performs stereo porjection from image space in 3D space
        Args:
            uVector (np.array): vector of image coordinates along the image width
            vVector (np.array): vector of image coordinates along the image height
            dVector (np.array): vector of disparity values corresponding to the image coordinates
            qMatrix (np.array): Q marix for stereo projection
        """
        w = (qMatrix[3, 2] * dVector) + qMatrix[3, 3]
        x = (uVector * qMatrix[0, 0] + qMatrix[0, 3]) / w
        y = (vVector * qMatrix[1, 1] + qMatrix[1, 3]) / w
        z =  qMatrix[2, 3] / w
        xyz = np.column_stack((x,y,z))
        return xyz