import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dartpy as dart

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    # input data porcessing
    from src.sensing.preProcessing import PreProcessing
    from src.sensing.dataHandler import DataHandler

    # topology extraction
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.localization.downsampling.l1median.l1Median import L1Median

    # model generation
    from src.simulation.bdlo import BranchedDeformableLinearObject

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise
global dataHandler
global evalConfig
global preProcessor

# results
results = {
    "preprocessing": [],
    "topologyExtraction": [],
    "localization": [],
    "tracking": [],
}

# visualization
visControl = {
    "preprocessing": {"vis": False, "block": False},
    "somResult": {"vis": False, "block": False},
    "extractedTopology": {"vis": False, "block": False},
    "generatedModel": {"vis": True, "block": True},
}
saveControl = {
    "parentDirectory": "data/eval/experiments/",
    "folderName": "20230509_ManipulationSequence_ArenaWireHarenss_Manual",
}
loadControl = {
    "parentDirectory": {
        "paths": ["data/darus_data_download/data/"],
        "index": 0,
    },
    "folderName": {
        "paths": [
            "20230508_174656_arenawireharness_manipulationsequence_manual/20230508_174656_ArenaWireHarness_ManipulationSequence_Manual/",
            "20230510_175759_YShape/",
            "20230510_190017_Partial/",
            "20230510_175016_singledlo/",
        ],
        "index": 3,
    },
    "initFile": {
        "index": 0,
    },
}


def getDataSetFileNames(dataSetFolderPath):
    dataSetFileNames = []
    for file in os.listdir(dataSetFolderPath + "/data"):
        if file.endswith("rgb.png"):
            dataSetFileNames.append(os.path.join("/mydir", file))
    return dataSetFileNames


def setupEvaluation():
    """sets up the evaluation
    reading the evaluation config, setting up and setting up the data handler
    """
    global dataHandler
    global evalConfig
    # read eval config
    evalConfigPath = os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/"
    evalConfigFiles = ["/evalConfig.json"]
    loadPath = (
        loadControl["parentDirectory"]["paths"][loadControl["parentDirectory"]["index"]]
        + loadControl["folderName"]["paths"][loadControl["folderName"]["index"]]
    )
    savePath = saveControl["parentDirectory"] + saveControl["folderName"]
    dataHandler = DataHandler(
        defaultLoadFolderPath=loadPath, defaultSaveFolderPath=savePath
    )
    evalConfig = dataHandler.loadFromJson(evalConfigPath + evalConfigFiles[0])
    preprocessingParameters = evalConfig["preprocessingParameters"]
    topologyExtractionParameters = evalConfig["topologyExtractionParameters"]
    return (
        preprocessingParameters,
        topologyExtractionParameters,
    )


def preprocessDataSet(dataSetFolder, dataSetFileName, preprocessingParameters):
    global preProcessor
    preProcessor = PreProcessing(
        defaultLoadFolderPath=dataSetFolder,
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

    results["preprocessing"].append(
        {
            "dataSetPath": dataSetFolder,
            "dataSetFileName": dataSetFileName,
            "robotBaseCoordinates": {
                "image": robotBase_inImageCoordinates,
                "3D": cameraCenter_inRobotBaseCoordinates,
                "cam": robotBase_inCameraCoordinates,
            },
            "cameraCoordinates": {
                "3D": cameraCenter_inRobotBaseCoordinates,
                "cam": np.array([0, 0, 0]),
            },
            "pointCloud": {
                "inliers": inliers,
                "inlierColors": inlierColors,
                "outliers": inlierColors,
                "outlierColors": outlierColors,
            },
        }
    )
    return inliers, inlierColors


def topologyExtraction(pointCloud, topologyExtractionParameters):
    Y = pointCloud[0]
    # topology extraction
    topologyExtraction = TopologyExtraction(
        Y=Y,
        somParameters=topologyExtractionParameters["somParameters"],
        l1Parameters=topologyExtractionParameters["l1Parameters"],
    )
    reducedPointSet = Y
    # reducedPointSet = topologyExtraction.reducePointSetL1(reducedPointSet)
    reducedPointSet = topologyExtraction.reducePointSetSOM(reducedPointSet)
    extractedTopology = topologyExtraction.extractTopology(reducedPointSet)

    # Visualization
    if visControl["somResult"]["vis"]:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plotPointSet(ax=ax, X=topologyExtraction.Y, color=[0, 0, 0], size=1)
        plotPointSet(
            ax=ax, X=topologyExtraction.reducedPointSetsSOM[0], color=[1, 0, 0], size=20
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal3D(ax)
        plt.show(block=visControl["somResult"]["block"])

    if visControl["extractedTopology"]["vis"]:
        # 2D image
        rgbImage = dataHandler.loadNumpyArrayFromPNG(
            results["preprocessing"][0]["dataSetFileName"]
        )
        rgbImage_topology = rgbImage.copy()
        pointPairs_inCameraCoordinates = (
            preProcessor.transformPointsFromRobotBaseToCameraCoordinates(
                np.concatenate(pointPairs, axis=0)
            )
        )
        U, V, D = preProcessor.inverseStereoProjection(
            pointPairs_inCameraCoordinates, preProcessor.cameraParameters["qmatrix"]
        )
        i = 0
        while i <= len(U) - 1:
            cv2.line(
                rgbImage_topology,
                (U[i], V[i]),
                (U[i + 1], V[i + 1]),
                (0, 255, 0),
                5,
            )
            i += 2
        fig = plt.figure()
        fig.add_subplot()
        plt.imshow(rgbImage_topology)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        pointPairs = extractedTopology.getAdjacentPointPairs()
        leafNodeIndices = extractedTopology.getLeafNodeIndices()
        pointPairs_inCamCoordinates = []
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
        plotPointSet(ax=ax, X=extractedTopology.X, color=[1, 0, 0], size=30)
        plotPointSet(ax=ax, X=extractedTopology.X, color=[1, 0, 0], size=20)
        plotPointSet(
            ax=ax,
            X=extractedTopology.X[leafNodeIndices, :],
            color=[1, 0, 0],
            size=50,
            alpha=0.4,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal3D(ax)
        plt.show(block=visControl["extractedTopology"]["block"])

    results["topologyExtraction"].append(
        {
            "reducedPointSet": extractedTopology.X,
            "reducedPointSetsSOM": topologyExtraction.reducedPointSetsSOM,
            "reducedPointSetsL1": topologyExtraction.reducedPointSetsL1,
            "extractedFeatureMatrix": topologyExtraction.extractedFeatureMatrix,
        }
    )
    return extractedTopology


def modelGeneration():
    global dataHandler
    # load model
    modelInfo = dataHandler.loadModelParameters("model.json")
    branchSpecs = list(modelInfo["branchSpecifications"].values())
    bdloModel = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": modelInfo["topologyModel"], "branchSpecs": branchSpecs}
    )
    # bdloModel.setBranchRootDof(1, 0, np.pi * 3 / 4)
    # bdloModel.setBranchRootDofs(2, np.array([0, 0, 0]))
    # bdloModel.setBranchRootDofs(3, np.array([-np.pi * 3 / 4, 0, 0]))
    # bdloModel.setBranchRootDofs(4, np.array([0, 0, 0]))
    # bdloModel.setBranchRootDofs(5, np.array([np.pi / 4, 0, 0]))
    # bdloModel.setBranchRootDofs(6, np.array([0, 0, 0]))

    if visControl["generatedModel"]["vis"]:
        world = dart.simulation.World()
        node = dart.gui.osg.WorldNode(world)
        # Create world node and add it to viewer
        viewer = dart.gui.osg.Viewer()
        viewer.addWorldNode(node)

        # add skeleton
        world.addSkeleton(bdloModel.skel)

        # Grid settings
        grid = dart.gui.osg.GridVisual()
        grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
        grid.setOffset([0, 0, 0])
        viewer.addAttachment(grid)

        viewer.setUpViewInWindow(0, 0, 1200, 900)
        viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
        if visControl["generatedModel"]["block"]:
            viewer.run()
        else:
            viewer.frame()


if __name__ == "__main__":
    # setup
    (preprocessingParameters, topologyExtractionParameters) = setupEvaluation()

    # choose file for initialization
    initDataSetFileName = dataHandler.getDataSetFileName_RBG(
        loadControl["initFile"]["index"]
    )
    # preprocessing
    pointCloud = preprocessDataSet(
        dataHandler.defaultLoadFolderPath, initDataSetFileName, preprocessingParameters
    )
    extractedTopology = topologyExtraction(pointCloud, topologyExtractionParameters)

    # model generation

    # TODO initialLocalization
    modelGeneration()
    # TODO tracking
