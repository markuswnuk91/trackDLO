import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dartpy as dart
from scipy.spatial import distance_matrix
from functools import partial

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

    # initial localization
    from src.localization.bdloLocalization import (
        BDLOLocalization,
    )

    # tracking
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart

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
    "preprocessing": {"vis": True, "block": False},
    "somResult": {"vis": True, "block": False},
    "extractedTopology": {"vis": True, "block": False},
    "generatedModel": {"vis": False, "block": False},
    "initialLocalization": {"vis": True, "block": True},
    "tracking": {"vis": True, "block": True},
}
saveControl = {
    "defaultPath": "data/eval/20230516_Test/",
    "preprocessing": {"save": False, "path": "data/eval/20230516_Test/Preprocessing/"},
    "localizaiton": {"save": False, "path": "data/eval/20230516_Test/Localization/"},
}

loadControl = {
    "dataSetPaths": [
        "data/darus_data_download/data/20230518_roboticwireharnessmounting/20230518_RoboticWireHarnessMounting/20230518_170955_YShape/",
        "data/darus_data_download/data/20230517_093521_manipulationsequence_manual_labeled_singledlo/20230517_093521_ManipulationSequence_manual_labeled_SingleDLO/",
    ],
    "dataSetToLoad": 1,
    "fileToLoad": 150,
    "parentDirectory": {
        "paths": [
            "data/darus_data_download/data/",
            "data/acquiredData/20230511_Configurations_Static_Overlap3D/",
        ],
        "index": 0,
    },
    "folderName": {
        "paths": [
            "20230508_174656_arenawireharness_manipulationsequence_manual/20230508_174656_ArenaWireHarness_ManipulationSequence_Manual/",
            "20230510_175759_YShape/",
            "20230510_190017_Partial/",
            "20230510_175016_singledlo/",
            "20230511_130114_Arena/",
            "20230511_112944_YShape/",
            "20230511_105435_Partial/",
        ],
        "index": 0,
    },
    "initFile": "20230508_174836529458_image_rgb.png",
}


def setupVisualization(dim):
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    elif dim <= 2:
        fig = plt.figure()
        ax = fig.add_subplot()
    return fig, ax


def setupVisualizationCallback(classHandle):
    fig2D, ax2D = setupVisualization(2)
    fig3D, ax3D = setupVisualization(3)
    if saveControl["localizaiton"]["save"]:
        savePath = saveControl["localization"]["path"]
    else:
        savePath = None
    return partial(
        visualizationCallback,
        fig2D,
        ax2D,
        fig3D,
        ax3D,
        classHandle,
        savePath=savePath,
    )


def visualizationCallback(
    fig2D,
    ax2D,
    fig3D,
    ax3D,
    classHandle,
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")

    # 2D image
    ax2D.cla()
    jointPositions = classHandle.templateTopology.getAdjacentPointPairs()
    rgbImage = dataHandler.loadNumpyArrayFromPNG(
        results["preprocessing"][0]["dataSetFileName"]
    )
    rgbImage_topology = rgbImage.copy()
    jointPositions_inCameraCoordinates = (
        preProcessor.transformPointsFromRobotBaseToCameraCoordinates(
            np.concatenate(jointPositions, axis=0)
        )
    )
    U, V, D = preProcessor.inverseStereoProjection(
        jointPositions_inCameraCoordinates, preProcessor.cameraParameters["qmatrix"]
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
    ax2D.imshow(rgbImage_topology)
    if savePath is not None:
        fig2D.savefig(
            savePath + fileName + "_" + str(classHandle.iter) + ".png",
            bbox_inches="tight",
        )

    # 3D Image
    ax3D.cla()
    plotPointSets(
        ax=ax3D,
        X=classHandle.X,
        Y=classHandle.YTarget,
        ySize=10,
        xSize=10,
    )
    for i, y in enumerate(classHandle.YTarget):
        plotLine(
            ax=ax3D,
            pointPair=np.vstack(((classHandle.C @ classHandle.X)[i], y)),
            color=[1, 0, 0],
            alpha=0.3,
        )
    plt.draw()
    plt.pause(0.01)
    return


def setupVisualizationCallbackTracking(classHandle):
    fig, ax = setupVisualization(classHandle.Y.shape[1])
    return partial(
        visualizationCallbackTracking,
        fig,
        ax,
        classHandle,
        savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/bldoReconstruction/test/",
    )


def visualizationCallbackTracking(
    fig,
    ax,
    classHandle,
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    ax.cla()
    plotPointSets(
        ax=ax,
        X=classHandle.T,
        Y=classHandle.Y,
        ySize=3,
        xSize=10,
    )
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.1)


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
    loadPath = loadControl["dataSetPaths"][loadControl["dataSetToLoad"]]
    savePath = saveControl["defaultPath"]
    dataHandler = DataHandler(
        defaultLoadFolderPath=loadPath, defaultSaveFolderPath=savePath
    )
    evalConfig = dataHandler.loadFromJson(evalConfigPath + evalConfigFiles[0])
    preprocessingParameters = evalConfig["preprocessingParameters"]
    topologyExtractionParameters = evalConfig["topologyExtractionParameters"]
    localizationParameters = evalConfig["localizationParameters"]
    trackingParameters = evalConfig["trackingParameters"]
    return (
        preprocessingParameters,
        topologyExtractionParameters,
        localizationParameters,
        trackingParameters,
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
    reducedPointSet = topologyExtraction.reducePointSetL1(reducedPointSet)
    reducedPointSet = topologyExtraction.reducePointSetSOM(reducedPointSet)
    reducedPointSet = topologyExtraction.pruneDuplicatePoints(
        reducedPointSet, topologyExtractionParameters["pruningThreshold"]
    )
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
        pointPairs = extractedTopology.getAdjacentPointPairs()
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
        leafNodeIndices = extractedTopology.getLeafNodeIndices()
        for pointPair in pointPairs:
            stackedPair = np.stack(pointPair)
            plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
        plotPointSet(
            ax=ax,
            X=extractedTopology.X,
            color=[0, 0, 1],
            size=30,
            alpha=0.4,
        )
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
    return bdloModel


def initialLocalization(
    pointCloud, extractedTopology, bdloModel, localizationParameters
):
    localCoordinateSamples = np.linspace(
        0,
        1,
        localizationParameters["numLocalCoordinateSamples"],
    )
    Y = pointCloud[0]
    localization = BDLOLocalization(
        **{
            "Y": Y,
            "S": localCoordinateSamples,
            "templateTopology": bdloModel,
            "extractedTopology": extractedTopology,
        }
    )

    if visControl["initialLocalization"]["vis"]:
        visualizationCallback = setupVisualizationCallback(localization)
        localization.registerCallback(visualizationCallback)
    result = localization.reconstructShape(numIter=localizationParameters["numIter"])
    qInit = result.x
    return qInit


def tracking(Y, bdloModel, qInit, trackingParameters):
    kinematicModel = KinematicsModelDart(bdloModel.skel.clone())
    B = []
    for i in range(0, bdloModel.getNumBranches()):
        B.append(bdloModel.getBranchBodyNodeIndices(i))
        KinematicsPreservingRegistration4BDLO
    kinematicModel.skel.setPositions(qInit)
    Dof = qInit.shape[0]
    stiffnessMatrix = np.eye(Dof)
    stiffnessMatrix[3:6, 3:6] = np.zeros((3, 3))
    reg = KinematicsPreservingRegistration4BDLO(
        qInit=qInit,
        q0=np.zeros(Dof),
        Y=Y,
        model=kinematicModel,
        B=B,
        stiffnessMatrix=stiffnessMatrix,
        **trackingParameters,
    )
    if visControl["tracking"]["vis"]:
        visualizationCallbackTracking = setupVisualizationCallbackTracking(reg)
        qHat = reg.register(visualizationCallbackTracking)
    else:
        qHat.register()
    return qHat


if __name__ == "__main__":
    # setup
    (
        preprocessingParameters,
        topologyExtractionParameters,
        localizationParameters,
        trackingParameters,
    ) = setupEvaluation()

    # choose file for initialization
    # if loadControl["initFile"].isnumeric():
    #     initDataSetFileName = dataHandler.getDataSetFileName_RBG(
    #         loadControl["initFile"]
    #     )
    # else:
    #     initDataSetIndex = dataHandler.getDataSetIndexFromFileName(
    #         loadControl["initFile"]
    #     )
    #     initDataSetFileName = dataHandler.getDataSetFileName_RBG(initDataSetIndex)

    initDataSetFileName = dataHandler.getDataSetFileNames()[loadControl["fileToLoad"]]
    # preprocessing
    pointCloud = preprocessDataSet(
        dataHandler.defaultLoadFolderPath, initDataSetFileName, preprocessingParameters
    )
    # model generation
    bdloModel = modelGeneration()

    # topology extraction
    extractedTopology = topologyExtraction(pointCloud, topologyExtractionParameters)

    # TODO initialLocalization
    qInit = initialLocalization(
        pointCloud, extractedTopology, bdloModel, localizationParameters
    )

    # TODO tracking
    qHat = tracking(pointCloud[0], bdloModel, qInit, trackingParameters)
    print(qHat)
