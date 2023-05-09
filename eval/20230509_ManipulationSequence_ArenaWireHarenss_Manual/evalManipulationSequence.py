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


# visualization
visControl = {"preprocessing": {"vis": True, "block": True}}
saveControl = {
    "parentDirectory": "data/eval/experiments/",
    "folderName": "20230509_ManipulationSequence_ArenaWireHarenss_Manual",
}
dataSetControl = {"folderNumber": 0, "fileNumber": 0}


def setupEvaluation():
    """sets up the evaluation
    reading the evaluation config, setting up and setting up the data handler
    """
    # read eval config
    evalConfigPath = os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/"
    evalConfigFiles = ["/evalConfig.json"]
    savePath = saveControl["parentDirectory"] + saveControl["folderName"]
    dataHandler = DataHandler(savePath)
    evalConfig = dataHandler.loadFromJson(evalConfigPath + evalConfigFiles[0])
    preprocessingParameters = evalConfig["preprocessingParameters"]
    return dataHandler, evalConfig, preprocessingParameters


def preprocessDataSet(dataSetPath, dataSetFileName, preprocessingParameters):
    preProcessor = PreProcessing(
        defaultLoadFolderPath=dataSetPath,
        hsvFilterParameters=preprocessingParameters["hsvFilterParameters"],
        roiFilterParameters=preprocessingParameters["roiFilterParameters"],
    )
    rgbImage, disparityMap = preProcessor.loadStereoDataSet_FromRGB(dataSetFileName)
    # point cloud generation
    points, colors = preProcessor.calculatePointCloudFiltered_2D_3D(
        rgbImage, disparityMap
    )
    # downsampling
    points, colors = preProcessor.downsamplePointCloud_nthElement(
        (points, colors),
        preprocessingParameters["downsamplingParameters"]["nthElement"],
    )

    # bounding box filter in camera coodinate system
    inliers, inlierColors = preProcessor.getInliersFromBoundingBox(
        (points, colors),
        preprocessingParameters["cameraCoordinateBoundingBoxParameters"],
    )
    outliers, outlierColors = preProcessor.getOutliersFromBoundingBox(
        (points, colors),
        preprocessingParameters["cameraCoordinateBoundingBoxParameters"],
    )

    # transfrom points in robot coodinate sytem
    inliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(inliers)
    outliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(outliers)
    points = np.vstack((inliers, outliers))
    colors = np.vstack((inlierColors, outlierColors))

    # bounding box filter in robot coordinate system
    inliers, inlierColors = preProcessor.getInliersFromBoundingBox(
        (points, colors),
        preprocessingParameters["robotCoordinateBoundingBoxParameters"],
    )
    outliers, outlierColors = preProcessor.getOutliersFromBoundingBox(
        (points, colors),
        preprocessingParameters["robotCoordinateBoundingBoxParameters"],
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
    if visControl["preprocessing"]["vis"]:
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
        plt.show(block=visControl["preprocessing"]["block"])
    return inliers, inlierColors


if __name__ == "__main__":
    # setup
    dataHandler, evalConfig, preprocessingParameters = setupEvaluation()
    # path managment
    dataSetPath = evalConfig["loadPathInfo"]["folderPaths"][
        dataSetControl["folderNumber"]
    ]
    dataSetFileName = evalConfig["loadPathInfo"]["fileNames"][
        dataSetControl["fileNumber"]
    ]
    # preprocessing
    pointCloud, pointCloudColors = preprocessDataSet(
        dataSetPath, dataSetFileName, preprocessingParameters
    )
    print(pointCloud)

    # TODO initialization

    # TODO tracking
