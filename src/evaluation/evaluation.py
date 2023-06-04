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

    # data loading functions
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

    def getVisualizationCallback(
        self,
        classHandle,
        visualizationFunction=None,
        fig=None,
        ax=None,
        dim=None,
        *args,
        **kwargs
    ):
        if dim is None:
            try:
                dim = classHandle.Y.shape[1]
            except:
                dim = 3
        fig, ax = self.setupFigure(fig, ax, dim)
        if visualizationFunction is None:
            visCallback = self.setupVisualizationCallback(
                self.standardVisualizationFunctions,
                fig,
                ax,
                classHandle,
                *args,
                **kwargs,
            )
        else:
            visCallback = self.setupVisualizationCallback(
                visualizationFunction,
                fig,
                ax,
                classHandle,
                *args,
                **kwargs,
            )
        return visCallback

    def setupFigure(self, fig=None, ax=None, dim=None):
        if dim is None:
            dim == 3
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
        return fig, ax

    def setupVisualizationCallback(
        self, visFunction, fig, ax, classHandle, *args, **kwargs
    ):
        return partial(
            visFunction,
            fig,
            ax,
            classHandle,
            *args,
            **kwargs,
        )

    def standardVisualizationFunctions(self, fig, ax, classHandle, *args, **kwargs):
        # determine type of classhandle
        if type(classHandle) == SelfOrganizingMap:
            ax.cla()
            plotPointSets(
                ax=ax,
                X=classHandle.T,
                Y=classHandle.Y,
                ySize=1,
                xSize=30,
                xColor=[1, 0, 0],
                yColor=[0, 0, 0],
            )
            set_axes_equal(ax)
            plt.draw()
            plt.pause(0.1)
        elif (
            (type(classHandle) == CoherentPointDrift)
            or (type(classHandle) == StructurePreservedRegistration)
            or (type(classHandle) == KinematicsPreservingRegistration)
        ):
            ax.cla()
            plotPointSets(
                ax=ax,
                X=classHandle.T,
                Y=classHandle.Y,
                ySize=1,
                xSize=30,
                xColor=[1, 0, 0],
                yColor=[0, 0, 0],
            )
            set_axes_equal(ax)
            plt.draw()
            plt.pause(0.1)
        else:
            raise NotImplementedError

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
