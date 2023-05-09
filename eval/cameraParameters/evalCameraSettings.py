import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.sensing.preProcessing import PreProcessing
    from src.sensing.dataHandler import DataHandler
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

vis = True

evalConfigPath = os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/"
evalConfigFiles = ["/evalConfig.json"]

fileNameNumber = 1 # index of the file in the filename list in evalConfig
dataHandler = DataHandler("data/eval/experiments/20230508_111645_testCameraSettings")
evalConfig = dataHandler.loadFromJson(evalConfigPath + evalConfigFiles[0])
evalParameters = evalConfig["preprocessingParameters"]


def testPointCloudExtraction():
    preProcessor = PreProcessing(
        defaultLoadFolderPath=evalConfig["loadPathInfo"]["folderPaths"][0],
        hsvFilterParameters=evalParameters["hsvFilterParameters"],
        roiFilterParameters=evalParameters["roiFilterParameters"],
    )
    rgbImage, disparityMap = preProcessor.loadStereoDataSet_FromRGB(
        evalConfig["loadPathInfo"]["fileNames"][fileNameNumber]
    )
    # point cloud generation
    points, colors = preProcessor.calculatePointCloudFiltered_2D_3D(
        rgbImage, disparityMap
    )
    # downsampling
    points, colors = preProcessor.downsamplePointCloud_nthElement((points, colors), 10)

    # bounding box filter in camera coodinate system
    inliers, inlierColors = preProcessor.getInliersFromBoundingBox(
        (points, colors), evalParameters["cameraCoordinateBoundingBoxParameters"]
    )
    outliers, outlierColors = preProcessor.getOutliersFromBoundingBox(
        (points, colors), evalParameters["cameraCoordinateBoundingBoxParameters"]
    )

    # transfrom points in robot coodinate sytem
    inliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(inliers)
    outliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(outliers)
    points = np.vstack((inliers, outliers))
    colors = np.vstack((inlierColors, outlierColors))

    # bounding box filter in robot coordinate system
    inliers, inlierColors = preProcessor.getInliersFromBoundingBox(
        (points, colors), evalParameters["robotCoordinateBoundingBoxParameters"]
    )
    outliers, outlierColors = preProcessor.getOutliersFromBoundingBox(
        (points, colors), evalParameters["robotCoordinateBoundingBoxParameters"]
    )
    outlierColors[:, :] = np.array([0.5, 0.5, 0.5])
    # camera center coordinates
    cameraCenter_inRobotBaseCoordinates = np.linalg.inv(
        preProcessor.calibrationParameters["T_Camera_To_RobotBase"]
    )[:3, 3]
    # robot base coordinates
    robotBase_inRobotBaseCoordinates = np.array([0, 0, 0])
    robotBase_inCameraCoordinates = preProcessor.calibrationParameters[
        "T_Camera_To_RobotBase"
    ][:3, 3]
    inverseStereoPorjectionResult = preProcessor.inverseStereoProjection(
        robotBase_inCameraCoordinates, preProcessor.cameraParameters["qmatrix"]
    )
    robotBase_inImageCoordinates = (
        inverseStereoPorjectionResult[0][0],
        inverseStereoPorjectionResult[1][0],
    )
    # Visualization
    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotPointSet(ax=ax, X=inliers, color=inlierColors, size=1)
        plotPointSet(ax=ax, X=outliers, color=outlierColors, size=1)
        plotPoint(
            ax=ax, x=cameraCenter_inRobotBaseCoordinates, size=50, color=[1, 0, 0]
        )
        plotPoint(ax=ax, x=robotBase_inRobotBaseCoordinates, size=50, color=[0, 0, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show(block=False)
        rgbImage_WithRobotBase = rgbImage.copy()
        cv2.circle(
            rgbImage_WithRobotBase, robotBase_inImageCoordinates, 30, (0, 0, 255), 5
        )  # draw cicle
        fig = plt.figure()
        fig.add_subplot()
        plt.imshow(rgbImage_WithRobotBase)
        plt.show(block=True)


if __name__ == "__main__":
    testPointCloudExtraction()
