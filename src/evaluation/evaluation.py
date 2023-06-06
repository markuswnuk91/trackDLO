import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dartpy as dart
from scipy.spatial import distance_matrix
from functools import partial
import pickle

try:
    sys.path.append(os.getcwd().replace("/src/evaluation", ""))
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
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )

    # model generation
    from src.simulation.bdlo import BranchedDeformableLinearObject

    # initial localization
    from src.localization.bdloLocalization import (
        BDLOLocalization,
    )

    # tracking
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import KinematicsPreservingRegistration
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart

    # visualization
    from src.visualization.plot3D import *

except:
    print("Imports for evaluation class failed.")
    raise


class Evaluation(object):
    def __init__(self, configFilePath, *args, **kwargs):
        self.configFilePath = configFilePath
        self.dataHandler = DataHandler()
        self.config = self.dataHandler.loadFromJson(self.configFilePath)
        self.results = []
        self.currentDataSetLoadPath = None
        self.currentLoadFileIdentifier = None

    # file handling
    def setDefaultLoadPathDataSet(self, dataSetFolderPath):
        self.currentDataSetLoadPath = dataSetFolderPath

    def setDefaultLoadFileIdentifier(self, fileIdentifier):
        self.currentLoadFileIdentifier = fileIdentifier

    def getFileName(self, fileIdentifier, dataSetFolderPath):
        fileName = self.dataHandler.getFileNameFromNameOrIndex(
            fileIdentifier, dataSetFolderPath
        )
        return fileName

    def getFilePath(self, fileIdentifier, dataSetFolderPath):
        return (
            dataSetFolderPath
            + "data/"
            + self.getFileName(fileIdentifier, dataSetFolderPath)
        )

    def getNumImageSetsInDataSet(self, dataSetFolderPath=None):
        if dataSetFolderPath is None:
            dataSetFolderPath = self.currentDataSetLoadPath
        else:
            self.currentDataSetLoadPath = dataSetFolderPath
        return self.dataHandler.getNumImageSetsInDataSet(dataSetFolderPath)

    def saveResults(
        self,
        folderPath,
        fileName=None,
        results=None,
        generateUniqueID=True,
        method="pickle",
    ):
        results = self.results if results is None else results
        if fileName is None and generateUniqueID:
            fileName = self.dataHandler.generateIdentifier(MS=False) + "_" + "results"
        elif fileName is None and not generateUniqueID:
            fileName = "results"
        elif fileName is not None and generateUniqueID:
            fileName = self.dataHandler.generateIdentifier(MS=False) + "_" + fileName

        if method == "pickle":
            filePath = folderPath + fileName + ".pkl"
            with open(filePath, "wb") as f:
                pickle.dump(results, f)

        if method == "json":
            filePath = folderPath + fileName + ".pkl"
            jsonifiedResults = self.dataHandler.convertNumpyToLists(results)
            self.dataHandler.saveDictionaryAsJson(
                jsonifiedResults, folderPath, fileName
            )
        return filePath

    def loadResults(self, filePath):
        _, file_extension = os.path.splitext(filePath)
        if file_extension == ".pkl":
            with open(filePath, "rb") as f:
                results = pickle.load(f)
        return results

    def getLastLoadedDataPath(self):
        return self.getFilePath(
            self.currentLoadFileIdentifier, self.currentDataSetLoadPath
        )

    def getLastLoadedFileIdentifier(self):
        return self.currentLoadFileIdentifier

    def getLastLoadedDataSetPath(self):
        return self.currentDataSetLoadPath

    # data loading
    def getDataSet(self, fileIdentifier=None, dataSetFolderPath=None):
        if fileIdentifier is None:
            fileIdentifier = self.currentLoadFileIdentifier
        else:
            self.currentLoadFileIdentifier = fileIdentifier

        if dataSetFolderPath is None:
            dataSetFolderPath = self.currentDataSetLoadPath
        else:
            self.currentDataSetLoadPath = dataSetFolderPath

        fileIndex = self.dataHandler.getFileIndexFromNameOrIndex(
            fileIdentifier, dataSetFolderPath
        )
        dataSetFileName = self.dataHandler.getDataSetFileNames(dataSetFolderPath)[
            fileIndex
        ]
        dataSet = self.dataHandler.loadStereoDataSet(
            dataSetFileName, dataSetFolderPath=dataSetFolderPath
        )
        return dataSet

    def getPointCloud(
        self, fileIdentifier, dataSetFolderPath, segmentationMethod="standard"
    ):
        if fileIdentifier is None:
            fileIdentifier = self.currentLoadFileIdentifier
        else:
            self.currentLoadFileIdentifier = fileIdentifier

        if dataSetFolderPath is None:
            dataSetFolderPath = self.currentDataSetLoadPath
        else:
            self.currentDataSetLoadPath = dataSetFolderPath
        if segmentationMethod == "standard":
            parameters = self.config["preprocessingParameters"]
            preProcessor = PreProcessing(
                defaultLoadFolderPath=dataSetFolderPath,
                hsvFilterParameters=parameters["hsvFilterParameters"],
                roiFilterParameters=parameters["roiFilterParameters"],
            )
            # load data
            rgbImage, disparityMap = self.getDataSet(fileIdentifier, dataSetFolderPath)
            # point cloud generation
            points, colors = preProcessor.calculatePointCloudFiltered_2D_3D(
                rgbImage, disparityMap
            )
            # downsampling
            points, colors = preProcessor.downsamplePointCloud_nthElement(
                (points, colors),
                parameters["downsamplingParameters"]["nthElement"],
            )
            # bounding box filter in camera coodinate system
            inliers, inlierColors = preProcessor.getInliersFromBoundingBox(
                (points, colors),
                parameters["cameraCoordinateBoundingBoxParameters"],
            )
            # transfrom points in robot coodinate sytem
            inliers = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(
                inliers
            )
            return (inliers, inlierColors)
        else:
            raise NotImplementedError

    # model generation
    def generateModel(self, dataSetPath, numBodyNodes):
        modelInfo = self.dataHandler.loadModelParameters("model.json", dataSetPath)
        branchSpecs = list(modelInfo["branchSpecifications"].values())
        bdloModel = BranchedDeformableLinearObject(
            **{
                "adjacencyMatrix": modelInfo["topologyModel"],
                "branchSpecs": branchSpecs,
                "defaultNumBodyNodes": numBodyNodes,
            }
        )
        return bdloModel

    # topology extraction
    def extractTopology(
        self,
        pointSet,
        somParameters=None,
        l1Parameters=None,
        pruningThreshold=None,
        skeletonize=True,
        visualizeSOMIteration=False,
        visualizeSOMResult=False,
        visualizeL1Iterations=False,
        visualizeL1Result=False,
        visualizeExtractionResult=False,
        somCallback=None,
        l1Callback=None,
        block=False,
    ):
        # setup topology extraction
        Y = pointSet
        topologyExtraction = TopologyExtraction(
            Y=Y,
            somParameters=somParameters,
            l1Parameters=l1Parameters,
        )

        # setup visualization callbacks
        if visualizeSOMIteration and somCallback is not None:
            topologyExtraction.selfOrganizingMap.registerCallback(somCallback)
        elif visualizeSOMIteration and somCallback is None:
            visualizationCallback_SOM = self.getVisualizationCallback(
                topologyExtraction.selfOrganizingMap
            )
            topologyExtraction.selfOrganizingMap.registerCallback(
                visualizationCallback_SOM
            )
        if visualizeL1Iterations and l1Callback is not None:
            topologyExtraction.l1Median.registerCallback(l1Callback)
        elif visualizeL1Iterations and l1Callback is None:
            visualizationCallback_L1 = self.getVisualizationCallback(
                topologyExtraction.l1Median
            )
            topologyExtraction.l1Median.registerCallback(visualizationCallback_L1)

        # run data reduction
        reducedPointSet = Y
        if skeletonize:
            reducedPointSet = topologyExtraction.reducePointSetL1(reducedPointSet)
        reducedPointSet = topologyExtraction.reducePointSetSOM(reducedPointSet)
        reducedPointSet = topologyExtraction.pruneDuplicatePoints(
            reducedPointSet, pruningThreshold
        )
        # extract topology
        extractedTopology = topologyExtraction.extractTopology(reducedPointSet)

        # visualize results
        if visualizeSOMResult and somCallback is not None:
            somCallback(topologyExtraction.selfOrganizingMap)
            plt.show(block=False)
        elif visualizeSOMResult and somCallback is None:
            visualizationCallback_SOM(topologyExtraction.selfOrganizingMap)
            plt.show(block=False)
        if visualizeSOMResult and somCallback is not None:
            somCallback(topologyExtraction.selfOrganizingMap)
            plt.show(block=False)
        elif visualizeSOMResult and somCallback is None:
            visualizationCallback_SOM(topologyExtraction.selfOrganizingMap)
            plt.show(block=False)
        if visualizeExtractionResult:
            fix, ax = self.setupFigure()
            self.standardVisualizationFunction(fix, ax, extractedTopology)
            plt.show(block=False)
        if (
            visualizeL1Result or visualizeSOMResult or visualizeExtractionResult
        ) and block:
            plt.show(block=True)
        return extractedTopology, topologyExtraction

    # visualization
    def getVisualizationCallback(
        self,
        classHandle,
        callbackFunction=None,
        fig=None,
        ax=None,
        dim=None,
        pauseInterval=0.1,
        *args,
        **kwargs
    ):
        if dim is None:
            try:
                dim = classHandle.Y.shape[1]
            except:
                dim = 3
        fig, ax = self.setupFigure(fig, ax, dim)
        if callbackFunction is None:
            visCallback = self.setupVisualizationCallback(
                self.standardVisualizationCallback,
                fig,
                ax,
                classHandle,
                pauseInterval=pauseInterval,
                *args,
                **kwargs,
            )
        else:
            visCallback = self.setupVisualizationCallback(
                callbackFunction,
                fig,
                ax,
                classHandle,
                pauseInterval,
                *args,
                **kwargs,
            )
        return visCallback

    def setupFigure(self, fig=None, ax=None, dim=None):
        if dim is None:
            dim = 3
        elif dim > 3 or dim < 1:
            raise ValueError(
                "Dimension of plots can only be 2D or 3D. Obtained {} for desired number of dimensions".format(
                    dim
                )
            )
        if fig is None:
            fig = plt.figure()
        if ax is None and dim == 3:
            ax = fig.add_subplot(projection="3d")
        elif ax is None and dim <= 2:
            ax = fig.add_subplot()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig, ax

    def setupVisualizationCallback(
        self, visFunction, fig, ax, classHandle, pauseInterval=0.1, *args, **kwargs
    ):
        return partial(
            visFunction,
            fig,
            ax,
            classHandle,
            pauseInterval,
            *args,
            **kwargs,
        )

    def standardVisualizationCallback(
        self, fig, ax, classHandle, pauseInterval=0.1, *args, **kwargs
    ):
        ax.cla()
        self.standardVisualizationFunction(fig, ax, classHandle)
        plt.draw()
        plt.pause(pauseInterval)

    def standardVisualizationFunction(self, fig, ax, classHandle, *args, **kwargs):
        # determine type of classhandle
        if (type(classHandle) == SelfOrganizingMap) or (type(classHandle) == L1Median):
            plotPointSets(
                ax=ax,
                X=classHandle.T,
                Y=classHandle.Y,
                ySize=1,
                xSize=30,
                xColor=[1, 0, 0],
                yColor=[0, 0, 0],
            )
        elif (
            (type(classHandle) == CoherentPointDrift)
            or (type(classHandle) == StructurePreservedRegistration)
            or (type(classHandle) == KinematicsPreservingRegistration)
        ):
            plotPointSets(
                ax=ax,
                X=classHandle.T,
                Y=classHandle.Y,
                ySize=1,
                xSize=30,
                xColor=[1, 0, 0],
                yColor=[0, 0, 0],
            )
        elif type(classHandle) == MinimalSpanningTreeTopology:
            pointPairs = classHandle.getAdjacentPointPairs()
            leafNodeIndices = classHandle.getLeafNodeIndices()
            branchNodeIndices = classHandle.getBranchNodeIndices()
            for pointPair in pointPairs:
                stackedPair = np.stack(pointPair)
                plotPointSet(
                    ax=ax,
                    X=classHandle.X,
                    color=[0, 0, 1],
                    size=30,
                    alpha=0.4,
                )
                plotLine(ax, pointPair=stackedPair, color=[0, 0, 1])
                plotPointSet(
                    ax=ax,
                    X=classHandle.X[leafNodeIndices, :],
                    color=[1, 0, 0],
                    size=50,
                    alpha=0.4,
                )
        else:
            raise NotImplementedError
        set_axes_equal(ax)

    def showImage(self, fileIdentifier=None, dataSetFolderPath=None):
        if fileIdentifier is None:
            fileIdentifier = self.currentLoadFileIdentifier
        else:
            self.currentLoadFileIdentifier = fileIdentifier
        if dataSetFolderPath is None:
            dataSetFolderPath = self.currentDataSetLoadPath
        else:
            self.currentDataSetLoadPath = dataSetFolderPath

        rgb_image, _ = self.getDataSet(fileIdentifier, dataSetFolderPath)
        fileName = self.getFileName(fileIdentifier, dataSetFolderPath)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # convert to bgr for cv2
        cv2.imshow(
            "RGB image: " + fileName, cv2.resize(bgr_image, None, fx=0.25, fy=0.25)
        )
        cv2.waitKey(0)
