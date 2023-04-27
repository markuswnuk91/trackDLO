import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.preProcessing import PreProcessing
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

vis = True
folderPath = "data/acquired_data/20230426_133624_DataSet/"
fileName = "20230426_133625761810_image_rgb.png"
fileName_disp = "20230426_133625761810_map_disparity.npy"


hsvFilterParameters = {
    "hueMin": 53,
    "hueMax": 76,
    "saturationMin": 35,
    "saturationMax": 164,
    "valueMin": 52,
    "valueMax": 255,
}
roiFilterParameters = {
    "uMin" : 0.5,
    "uMax" : 0.95,
    "vMin" : 0.03,
    "vMax" : 0.87,
}

boundingBoxParameters = {
    "xMin" : 0,
    "xMax" : 1,
    "yMin" : 0,
    "yMax" : 1,
    "zMin" : 0,
    "zMax" : 1,
}

innerBoundingBoxParameters = {
    "xMin" : 0,
    "xMax" : 1,
    "yMin" : 0,
    "yMax" : 1,
    "zMin" : 0,
    "zMax" : 1,
}

def testPointCloudExtraction():
    preProcessor = PreProcessing(folderPath = folderPath, hsvFilterParameters = hsvFilterParameters, roiFilterParameters = roiFilterParameters , boundingBoxParameters= boundingBoxParameters)
    rgbImage, disparityMap = preProcessor.loadStereoDataSet_FromRGB(fileName)
    # point cloud generation
    points, colors = preProcessor.calculatePointCloudFiltered_2D_3D(rgbImage,disparityMap)
    #downsampling
    points, colors = preProcessor.downsamplePointCloud_nthElement((points, colors),100)
    # bounding box filter
    inliers, inlierColors = preProcessor.getInliersFromBoundingBox((points, colors),innerBoundingBoxParameters)
    outliers, outlierColors = preProcessor.getOutliersFromBoundingBox((points, colors),innerBoundingBoxParameters)
    outlierColors[:,:] = np.array([0.5,0.5,0.5])

    inliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(inliers)
    outliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(outliers)

    cameraCenter_inRobotBaseCoordinates = np.linalg.inv(preProcessor.calibrationParameters["T_Camera_To_RobotBase"])[:3,3]
    robotBase_inRobotBaseCoordinates = np.array([0,0,0])
    robotBase_inCameraCoordinates = preProcessor.calibrationParameters["T_Camera_To_RobotBase"][:3,3]
    inverseStereoPorjectionResult =  preProcessor.inverseStereoProjection(robotBase_inCameraCoordinates,preProcessor.cameraParameters["qmatrix"])
    robotBase_inImageCoordinates = (inverseStereoPorjectionResult[0][0], inverseStereoPorjectionResult[1][0])
    # Visualization
    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotPointSet(ax = ax, X=inliers, color = inlierColors, size=1)
        plotPointSet(ax = ax, X=outliers, color = outlierColors, size=1)
        plotPoint(ax = ax, x = cameraCenter_inRobotBaseCoordinates, size = 50, color = [1,0,0])
        plotPoint(ax=ax, x =robotBase_inRobotBaseCoordinates, size = 50, color = [0,0,1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show(block=False)
        rgbImage_WithRobotBase = rgbImage.copy()
        cv2.circle(rgbImage_WithRobotBase,robotBase_inImageCoordinates,30,(0, 0, 255),5) # draw cicle
        fig = plt.figure()
        fig.add_subplot()
        plt.imshow(rgbImage_WithRobotBase)
        plt.show(block=True)
        
if __name__ == "__main__":
    testPointCloudExtraction()