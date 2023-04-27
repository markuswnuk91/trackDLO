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

        return xyz, colors/255
    
    def stereoProjection(self, uVector,vVector,dVector,qMatrix):
        """performs stereo projection from image space in 3D space
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
    
    def inverseStereoProjection(self, points, qMatrix): 
        if len(points.shape)==1:
            points = np.expand_dims(points,axis=0)
        w = qMatrix[2, 3] / points[:,2]
        u = ((points[:,0] * w) - qMatrix[0, 3]) / qMatrix[0, 0]
        v = ((points[:,1] * w) - qMatrix[1, 3]) / qMatrix[1, 1]
        d = (w - qMatrix[3, 3]) / qMatrix[3, 2]
        return u.astype(int) , v.astype(int), d.astype(int)
    
    def getMaskFromBoundingBox(self, points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf, max_y=np.inf, min_z=-np.inf, max_z=np.inf):
        """ Compute a mask of a bounding box filter on the given points
        Method from: https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy
        Args:                       
            points: (n,3) array
                The array containing all the points's coordinates. Expected format:
                    array([
                        [x1,y1,z1],
                        ...,
                        [xn,yn,zn]])

            min_i, max_i: float
                The bounding box limits for each coordinate. If some limits are missing,
                the default values are -infinite for the min_i and infinite for the max_i.

        Returns:
            indexMask : boolean array
                The boolean mask of indices indicating wherever a point should be keeped or not.
                The size of the boolean mask will be the same as the number of given points.

        """
        bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
        bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
        bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

        indexMask = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

        return indexMask
    
    def filterPointsBoundingBox(self, points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf, max_y=np.inf, min_z=-np.inf, max_z=np.inf):
        """Determines inliers and outliers of a given point cloud and bounding box
        Args:
            points: (n,3) array
                The array containing all the points's coordinates. Expected format:
                    array([
                        [x1,y1,z1],
                        ...,
                        [xn,yn,zn]])

            min_i, max_i: float
                The bounding box limits for each coordinate. If some limits are missing,
                the default values are -infinite for the min_i and infinite for the max_i.
        Returns:
            filteredPoints : np. array
                The array of remaining points after applying the boxfilter
        """
        mask = self.getMaskFromBoundingBox(points, min_x, max_x, min_y, max_y, min_z, max_z)
        filteredPoints = points[mask,:]
        return filteredPoints
    
    def downsamplePointCloud_nthElement(self, pointCloud: tuple, nthElement):
        points = pointCloud[0]
        colors = pointCloud[1]
        points = points[::nthElement,:]
        colors = colors[::nthElement,:]
        return points, colors
    
    def transformPoints(self, points: np.array, transformationMatrix: np.array):
        transformedPoints = points @ transformationMatrix[:3,:3].T + transformationMatrix[:3,3]
        return transformedPoints