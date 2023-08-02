import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import dartpy as dart
from scipy.spatial import distance_matrix
from functools import partial
import pickle
from warnings import warn
import time

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
    from src.tracking.krcpd.krcpd import (
        KinematicRegularizedCoherentPointDrift,
    )
    from src.tracking.krcpd.krcpd4BDLO import (
        KinematicRegularizedCoherentPointDrift4BDLO,
    )

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
        self.results = {}
        self.resultLog = {
            "topologyExtraction": [],
            "localization": [],
            "initialization": [],
        }
        self.currentDataSetLoadPath = None
        self.currentLoadFileIdentifier = None

        # setup colormaps
        self.colorMaps = self.setupColorMaps()

        # variable to store loaded models
        self.generatedModels = []

    # ---------------------------------------------------------------------------
    # SETUP FUNCITONS
    # ---------------------------------------------------------------------------
    def setupColorMaps(self):
        colorMapDict = {}
        # set all colormaps to range from 0 to 1
        lowerLim = 0
        upperLim = 1
        norm = matplotlib.colors.Normalize(vmin=lowerLim, vmax=upperLim)  # Normalizer

        # add viridis colormap
        colorMap_viridis = matplotlib.colormaps["viridis"]
        scalarMappable_viridis = plt.cm.ScalarMappable(
            cmap=colorMap_viridis, norm=norm
        )  # creating ScalarMappable
        colorMapDict["viridis"] = scalarMappable_viridis
        return colorMapDict

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # FILE HANDLING FUNCTIONS
    def setDefaultLoadPathDataSet(self, dataSetFolderPath):
        self.currentDataSetLoadPath = dataSetFolderPath

    def setDefaultLoadFileIdentifier(self, fileIdentifier):
        self.currentLoadFileIdentifier = fileIdentifier

    def getFileName(self, fileIdentifier, dataSetFolderPath):
        fileName = self.dataHandler.getFileNameFromNameOrIndex(
            fileIdentifier, dataSetFolderPath
        )
        return fileName

    def getFileIdentifierFromFilePath(self, filePath):
        return filePath.split("/")[-1]

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
        promtOnSave=False,
        overwrite=True,
    ):
        if promtOnSave:
            key = input(
                'Saving results to: {}, with name {} as type {}.\n Press "y" to continue saving: '.format(
                    folderPath, fileName, method
                )
            )
        else:
            key = "y"
        if key == "y":
            results = self.results if results is None else results
            if fileName is None and generateUniqueID:
                fileName = (
                    self.dataHandler.generateIdentifier(MS=False) + "_" + "results"
                )
            elif fileName is None and not generateUniqueID:
                fileName = "results"
            elif fileName is not None and generateUniqueID:
                fileName = (
                    self.dataHandler.generateIdentifier(MS=False) + "_" + fileName
                )

            # create directory if it does not exist
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)

            if method == "pickle":
                filePath = folderPath + fileName + ".pkl"
                # check if file already exists
                if not overwrite and os.path.exists(filePath):
                    return None
                with open(filePath, "wb") as f:
                    pickle.dump(results, f)

            if method == "json":
                filePath = folderPath + fileName + ".json"
                # check if file already exists
                if not overwrite and os.path.exists(filePath):
                    return None
                jsonifiedResults = self.dataHandler.convertNumpyToLists(results)
                self.dataHandler.saveDictionaryAsJson(
                    jsonifiedResults, folderPath, fileName
                )
            return filePath
        else:
            print("Results were not saved.")

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

    def findCorrespondingEntryFromKeyValuePair(self, inputListOfDict, key, value):
        for entry in inputListOfDict:
            if entry[key] == value:
                return entry
        return None

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
                hsvPassThroughFilters=parameters["hsvPassThroughFilters"],
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
            # bounding box filter in robot coodinate system
            inliers, inlierColors = preProcessor.getInliersFromBoundingBox(
                (inliers, inlierColors),
                parameters["robotCoordinateBoundingBoxParameters"],
            )
            return (inliers, inlierColors)
        else:
            raise NotImplementedError

    def reprojectFrom3DRobotBase(self, coordinates3D, dataSetFolderPath):
        preProcessor = PreProcessing(defaultLoadFolderPath=dataSetFolderPath)
        coordinates3D_inCamera = (
            preProcessor.transformPointsFromRobotBaseToCameraCoordinates(coordinates3D)
        )
        U, V, D = preProcessor.inverseStereoProjection(
            coordinates3D_inCamera, preProcessor.cameraParameters["qmatrix"]
        )
        coordinates2D = np.vstack((U, V)).T
        return coordinates2D

    def getMarkerInformation(self, dataSetPath):
        markerInformation = self.dataHandler.loadMarkerInformation(
            fileName="model.json", folderPath=dataSetPath
        )
        return markerInformation

    def getMarkerBranchLocalCoordinates(self, dataSetPath):
        markerInformation = self.getMarkerInformation(dataSetPath)
        branchLocalCoordinates = []
        for i, markerInfo in enumerate(markerInformation):
            # make sure labels are provided in correct order
            if i != (markerInfo["number"] - 1):
                raise ValueError(
                    "Labels not in correct order. Expected label: {} but got label: {}.".format(
                        i, markerInfo["number"] - 1
                    )
                )
            else:
                branchIdx = markerInfo["branch"] - 1
                s = markerInfo["lenthFromBranchRootNode"] / markerInfo["branchLength"]
                branchLocalCoordinates.append((branchIdx, s))
        return branchLocalCoordinates

    # model generation
    def getModelParameters(self, dataSetPath, numBodyNodes=None):
        numBodyNodes = (
            self.config["modelGeneration"]["numSegments"]
            if numBodyNodes is None
            else numBodyNodes
        )
        modelInfo = self.dataHandler.loadModelParameters("model.json", dataSetPath)
        branchSpecs = list(modelInfo["branchSpecifications"].values())
        adjacencyMatrix = modelInfo["topologyModel"]
        modelParameters = {
            "modelInfo": modelInfo,
            "adjacencyMatrix": adjacencyMatrix,
            "branchSpecs": branchSpecs,
            "numBodyNodes": numBodyNodes,
        }
        return modelParameters

    def generateModel(self, modelParameters, cacheModel=False):
        bdloModel = BranchedDeformableLinearObject(
            **{
                "adjacencyMatrix": modelParameters["adjacencyMatrix"],
                "branchSpecs": modelParameters["branchSpecs"],
                "defaultNumBodyNodes": modelParameters["numBodyNodes"],
            }
        )
        if cacheModel:
            self.generatedModels.append(
                {"model": bdloModel, "parameters": modelParameters}
            )
        return bdloModel

    def getModel(self, dataSetPath, numBodyNodes=None):
        modelParameters = self.getModelParameters(dataSetPath, numBodyNodes=None)
        model = self.generateModel(modelParameters)
        return model, modelParameters

    # topology extraction
    def extractTopology(
        self,
        pointSet,
        somParameters=None,
        l1Parameters=None,
        pruningThreshold=None,
        skeletonize=True,
        visualizeSOMIterations=False,
        visualizeSOMResult=False,
        visualizeL1Iterations=False,
        visualizeL1Result=False,
        visualizeExtractionResult=False,
        somCallback=None,
        l1Callback=None,
        block=False,
        closeAfterRunning=True,
    ):
        # setup topology extraction
        Y = pointSet
        topologyExtraction = TopologyExtraction(
            Y=Y,
            somParameters=somParameters,
            l1Parameters=l1Parameters,
        )

        # setup visualization callbacks
        # l1
        if (visualizeL1Result or visualizeL1Iterations) and l1Callback is None:
            visualizationCallback_L1 = self.getVisualizationCallback(
                topologyExtraction.l1Median
            )
        else:
            visualizationCallback_L1 = l1Callback
        if visualizeL1Iterations:
            topologyExtraction.l1Median.registerCallback(visualizationCallback_L1)
        # som
        if (visualizeSOMResult or visualizeSOMIterations) and somCallback is None:
            visualizationCallback_SOM = self.getVisualizationCallback(
                topologyExtraction.selfOrganizingMap
            )
        else:
            visualizationCallback_SOM = somCallback
        if visualizeSOMIterations:
            topologyExtraction.selfOrganizingMap.registerCallback(
                visualizationCallback_SOM
            )
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
        if visualizeSOMResult:
            visualizationCallback_SOM(topologyExtraction.selfOrganizingMap)
            plt.show(block=False)
        if visualizeL1Result:
            visualizationCallback_L1(topologyExtraction.l1Median)
            plt.show(block=False)
        if visualizeExtractionResult:
            self.visualizeTopologyExtractionResult(
                topologyExtraction=topologyExtraction
            )
            plt.show(block=False)
        if (
            visualizeL1Result or visualizeSOMResult or visualizeExtractionResult
        ) and block:
            plt.show(block=True)
        if closeAfterRunning:
            plt.close("all")
        return extractedTopology, topologyExtraction

    def runTopologyExtraction(
        self,
        pointSet,
        somParameters=None,
        l1Parameters=None,
        pruningThreshold=None,
        skeletonize=True,
        visualizeSOMIterations=False,
        visualizeSOMResult=False,
        visualizeL1Iterations=False,
        visualizeL1Result=False,
        visualizeExtractionResult=False,
        somCallback=None,
        l1Callback=None,
        block=False,
        closeAfterRunning=True,
        logResults=True,
    ):
        somParameters = (
            self.config["topologyExtraction"]["somParameters"]
            if somParameters is None
            else somParameters
        )
        l1Parameters = (
            self.config["topologyExtraction"]["l1Parameters"]
            if l1Parameters is None
            else l1Parameters
        )
        pruningThreshold = (
            self.config["topologyExtraction"]["pruningThreshold"]
            if pruningThreshold is None
            else pruningThreshold
        )
        # extract topology
        extractedTopology, topologyExtraction = self.extractTopology(
            pointSet,
            somParameters=somParameters,
            l1Parameters=l1Parameters,
            pruningThreshold=pruningThreshold,
            skeletonize=skeletonize,
            visualizeSOMIterations=visualizeSOMIterations,
            visualizeSOMResult=visualizeSOMResult,
            visualizeL1Iterations=visualizeL1Iterations,
            visualizeL1Result=visualizeL1Result,
            visualizeExtractionResult=visualizeExtractionResult,
            somCallback=somCallback,
            l1Callback=l1Callback,
            closeAfterRunning=closeAfterRunning,
            block=block,
        )
        somResult = {
            "X": topologyExtraction.selfOrganizingMap.X,
            "Y": topologyExtraction.selfOrganizingMap.Y,
            "T": topologyExtraction.selfOrganizingMap.T,
            "runtimes": topologyExtraction.selfOrganizingMap.runTimes,
        }
        l1Result = {
            "X": topologyExtraction.l1Median.X,
            "Y": topologyExtraction.l1Median.Y,
            "T": topologyExtraction.l1Median.T,
            "runtimes": topologyExtraction.l1Median.runTimes,
        }
        extractedTopologyResult = {
            "X": extractedTopology.X,
            "featureMatrix": extractedTopology.featureMatrix,
        }
        topologyExtractionResult = {
            "som": somResult,
            "l1": l1Result,
            "result": extractedTopologyResult,
            "minimalSpanningTree": extractedTopology,
            "runtimes": {
                "topologyExtraction": topologyExtraction.runTimes,
                "l1": topologyExtraction.l1Median.runTimes,
                "som": topologyExtraction.selfOrganizingMap.runTimes,
            },
        }

        if logResults:
            self.resultLog["topologyExtraction"].append(topologyExtractionResult)

        return topologyExtractionResult, extractedTopology
        # -------------------------------------------------------------------------

    # Initial Localization
    # -------------------------------------------------------------------------
    def initialLocalization(
        self,
        pointSet,
        extractedTopology,
        bdloModelParameters,
        numSamples=10,
        numIterations=100,
        verbose=0,
        method="IK",
        jacobianDamping=None,
        visualizeCorresponanceEstimation=False,
        visualizeIterations=False,
        visualizeResult=False,
        visualizationCallback=None,
        block=False,
        closeAfterRunning=True,
    ):
        localCoordinateSamples = np.linspace(0, 1, numSamples)
        Y = pointSet
        bdloModel = self.generateModel(bdloModelParameters)

        localization = BDLOLocalization(
            **{
                "Y": Y,
                "S": localCoordinateSamples,
                "templateTopology": bdloModel,
                "extractedTopology": extractedTopology,
                "jacobianDamping": jacobianDamping,
            }
        )
        if visualizeIterations:
            visualizationCallback = self.getVisualizationCallback(localization)
            localization.registerCallback(visualizationCallback)

        # run inital localization
        qInit = localization.reconstructShape(
            numIter=numIterations, verbose=verbose, method=method
        )
        XInit, BInit, SInit = bdloModel.computeForwardKinematics(
            qInit, locations="center", returnBranchLocalCoordinates=True
        )
        # visualization
        if visualizeCorresponanceEstimation:
            fig, ax = self.setupFigure()
            self.standardVisualizationFunction(fig, ax, localization)
            plt.show(block=block)
        if closeAfterRunning:
            plt.close("all")
        return XInit, BInit, SInit, qInit, localization

    def runInitialLocalization(
        self,
        pointSet,
        extractedTopology,
        bdloModelParameters,
        numSamples=None,
        numIterations=None,
        verbose=None,
        method=None,
        jacobianDamping=None,
        visualizeCorresponanceEstimation=True,
        visualizeIterations=True,
        visualizeResult=True,
        visualizationCallback=None,
        block=False,
        closeAfterRunning=True,
        logResults=True,
    ):
        # if (dataSetPath is None and frame is None) and pointSet is None:
        #     raise ValueError(
        #         "Provide a path to a data set and the frame or a point cloud for which the initial localization should be performed."
        #     )
        # elif (dataSetPath is None and frame is None) and extractedTopology is None:
        #     raise ValueError(
        #         "Provide a path to a data set and the frame or a extracted topology for which the initial localization should be performed."
        #     )
        # elif (dataSetPath is None and frame is None) and bdloModel is None:
        #     raise ValueError(
        #         "Provide a path to a data set and the frame or a bdloModel for which the initial localization should be performed."
        #     )
        # elif dataSetPath is not None and frame is not None:
        #     pointSet = self.getPointCloud(frame, dataSetPath)
        #     bdloModel = self.generateModel(dataSetPath)
        #     _, extractedTopology = self.runTopologyExtraction(pointSet)
        numSamples = (
            self.config["localization"]["numSamples"]
            if numSamples is None
            else numSamples
        )
        numIterations = (
            self.config["localization"]["numIterations"]
            if numIterations is None
            else numIterations
        )
        verbose = self.config["localization"]["verbose"] if verbose is None else verbose
        method = self.config["localization"]["method"] if method is None else method
        jacobianDamping = (
            self.config["localization"]["jacobianDamping"]
            if jacobianDamping is None
            else jacobianDamping
        )

        XInit, BInit, SInit, qInit, localization = self.initialLocalization(
            pointSet=pointSet,
            extractedTopology=extractedTopology,
            bdloModelParameters=bdloModelParameters,
            numSamples=numSamples,
            numIterations=numIterations,
            verbose=verbose,
            method=method,
            jacobianDamping=jacobianDamping,
            visualizeCorresponanceEstimation=visualizeCorresponanceEstimation,
            visualizeIterations=visualizeIterations,
            visualizeResult=visualizeResult,
            visualizationCallback=visualizationCallback,
            block=block,
            closeAfterRunning=closeAfterRunning,
        )
        localizationResult = {
            "S": localization.S,
            "C": localization.C,
            "Y": localization.Y,
            "X": localization.X,
            "q": localization.q,
            "XLog": localization.XLog,
            "qLog": localization.qLog,
            "XInit": XInit,
            "qInit": qInit,
            "BInit": BInit,
            "SInit": SInit,
            "modelParameters": bdloModelParameters,
            "extractedTopology": localization.extractedTopology,
            "runtimes": localization.runTimes,
        }
        if logResults:
            self.resultLog["localization"].append(localizationResult)
        return localizationResult, localization

    def runInitialization(
        self,
        dataSetPath,
        frame,
        bdloModelParameters=None,
        somParameters=None,
        l1Parameters=None,
        pruningThreshold=None,
        skeletonize=True,
        visualize=True,
        visualizeSOMIterations=False,
        visualizeSOMResult=False,
        visualizeL1Iterations=False,
        visualizeL1Result=False,
        visualizeExtractionResult=False,
        somCallback=None,
        l1Callback=None,
        numSamples=None,
        numIterations=None,
        verbose=None,
        method=None,
        jacobianDamping=None,
        visualizeCorresponanceEstimation=True,
        visualizeIterations=True,
        visualizeResult=True,
        visualizationCallback=None,
        block=False,
        closeAfterRunningTopologyExtraction=False,
        closeAfterRunningLocalization=True,
        logResults=True,
    ):
        if visualize is False:
            visualizeSOMIterations = False
            visualizeSOMResult = False
            visualizeL1Iterations = False
            visualizeL1Result = False
            visualizeExtractionResult = False
            visualizeCorresponanceEstimation = False
            visualizeIterations = False
            visualizeResult = False

        pointCloudProcessingRuntime_start = time.time()
        pointCloud = self.getPointCloud(frame, dataSetPath)
        pointCloudProcessingRuntime_end = time.time()

        modelGenerationRuntime_start = time.time()
        bdloModelParameters = (
            self.getModelParameters(dataSetPath)
            if bdloModelParameters is None
            else bdloModelParameters
        )
        modelGenerationRuntime_end = time.time()

        pointSet = pointCloud[0]
        topologyExtractionResult, extractedTopology = self.runTopologyExtraction(
            pointSet,
            somParameters,
            l1Parameters,
            pruningThreshold,
            skeletonize,
            visualizeSOMIterations,
            visualizeSOMResult,
            visualizeL1Iterations,
            visualizeL1Result,
            visualizeExtractionResult,
            somCallback,
            l1Callback,
            block,
            closeAfterRunningTopologyExtraction,
        )
        initialLocalizationResult, _ = self.runInitialLocalization(
            pointSet,
            extractedTopology,
            bdloModelParameters,
            numSamples,
            numIterations,
            verbose,
            method,
            jacobianDamping,
            visualizeCorresponanceEstimation,
            visualizeIterations,
            visualizeResult,
            visualizationCallback,
            block,
            closeAfterRunningLocalization,
            logResults,
        )
        initializationResult = {
            "dataSetPath": dataSetPath,
            "fileName": self.getFileName(frame, dataSetPath),
            "filePath": self.getFilePath(frame, dataSetPath),
            "frame": frame,
            "pointCloud": pointCloud,
            "modelParameters": bdloModelParameters,
            "topologyExtraction": topologyExtractionResult,
            "localization": initialLocalizationResult,
            "runtimes": {
                "pointCloudProcessing": pointCloudProcessingRuntime_end
                - pointCloudProcessingRuntime_start,
                "modelGeneration": modelGenerationRuntime_end
                - modelGenerationRuntime_start,
                "topologyExtraction": topologyExtractionResult["runtimes"],
                "localization": initialLocalizationResult["runtimes"],
            },
        }
        if logResults:
            self.resultLog["initialization"].append(initializationResult)
        return initializationResult

    # -------------------------------------------------------------------------
    # TRACKING FUNCTIONS
    # -------------------------------------------------------------------------
    def runRegistration(
        self,
        registration,
        Y=None,
        X=None,
        T=None,
        checkConvergence=True,
        logTargets=True,
    ):
        if Y is not None:
            registration.Y = Y
        if X is not None:
            registration.X = X
        if T is not None:
            registration.T = T

        registrationResult = {}
        if logTargets:
            registrationResult["TLog"] = []
            # setup result callback
            logTargetsCallback = lambda: registrationResult["TLog"].append(
                registration.T.copy()
            )
        registration.register(
            checkConvergence=checkConvergence, customCallback=logTargetsCallback
        )

        # gather results
        registrationResult["X"] = registration.X.copy()
        registrationResult["Y"] = registration.Y.copy()
        registrationResult["T"] = registration.T.copy()

        # gather registration specific results
        if (
            type(registration) == CoherentPointDrift
            or type(registration) == StructurePreservedRegistration
        ):
            registrationResult["W"] = registration.W.copy()
            registrationResult["G"] = registration.G.copy()
        elif type(registration) == KinematicRegularizedCoherentPointDrift:
            registrationResult["W"] = registration.W.copy()
            registrationResult["G"] = registration.G.copy()
            registrationResult["q"] = registration.q.copy()
            registrationResult["Xreg"] = registration.Xreg.copy()

        return registrationResult

    def runTracking(
        self,
        dataSetPath,
        method,
        bdloModelParameters=None,
        model=None,
        startFrame=None,
        endFrame=None,
        frameStep=None,
        Y=None,
        XInit=None,
        qInit=None,
        S=None,
        B=None,
        locations="center",
        trackingParameters=None,
        visualize=True,
        visualizationCallback=None,
        checkConvergence=True,
        logTargets=True,
    ):
        # setup tracking problem
        if startFrame is None and Y is None:
            warn(
                "Provided neither frame nor point cloud to perform tracking on. Trying to use last localization result."
            )
            try:
                Y = self.resultLog["initialization"][-1]["pointCloud"][0]
            except:
                raise ValueError("No point set to perfrom registration on")
        bdloModelParameters = (
            self.getModelParameters(dataSetPath)
            if bdloModelParameters is None
            else bdloModelParameters
        )
        # setup the model model
        bdloModel = self.generateModel(bdloModelParameters) if model is None else model
        kinematicsModel = KinematicsModelDart(bdloModel.skel.clone())

        if startFrame is not None and Y is None:
            Y = self.getPointCloud(startFrame, dataSetPath)[0]
        if startFrame is None and Y is not None:
            Y = Y
        if startFrame is not None and Y is not None:
            warn(
                "Provided point cloud and frame. Using provided point cloud to continue"
            )
            Y = Y
        # XInit = (
        #     self.resultLog["initialization"][-1]["localization"]["XInit"]
        #     if XInit is None
        #     else XInit
        # )
        qInit = (
            self.resultLog["initialization"][-1]["localization"]["qInit"]
            if qInit is None
            else qInit
        )
        if XInit is None:
            XInit, B, S = bdloModel.computeForwardKinematics(
                qInit, locations=locations, returnBranchLocalCoordinates=True
            )
        else:
            if S is None or B is None:
                ValueError(
                    "Branch coordinates not specified for tracking. Please provide the branch coordinates corresponding to the initial confiugration."
                )
        endFrame = (
            self.getNumImageSetsInDataSet(dataSetPath) if endFrame is None else endFrame
        )
        frameStep = 1 if frameStep is None else frameStep
        if startFrame is not None and endFrame is not None:
            framesToTrack = list(range(startFrame, endFrame, frameStep))
        else:
            framesToTrack = []

        # setup results
        trackingResult = {}
        trackingResult["method"] = method
        trackingResult["registrations"] = []
        trackingResult["modelParameters"] = bdloModelParameters
        trackingResult["B"] = B
        trackingResult["S"] = S
        if method == "cpd":
            trackingParameters = (
                self.config["cpdParameters"]
                if trackingParameters is None
                else trackingParameters
            )
            reg = CoherentPointDrift(Y=Y, X=XInit, **trackingParameters)
            if visualize:
                if visualizationCallback is None:
                    visualizationCallback = self.getVisualizationCallback(reg)
                reg.registerCallback(visualizationCallback)
        elif method == "spr":
            trackingParameters = (
                self.config["sprParameters"]
                if trackingParameters is None
                else trackingParameters
            )
            reg = StructurePreservedRegistration(Y=Y, X=XInit, **trackingParameters)
            if visualize:
                if visualizationCallback is None:
                    visualizationCallback = self.getVisualizationCallback(reg)
                reg.registerCallback(visualizationCallback)
        elif method == "krcpd":
            trackingParameters = (
                self.config["krcpdParameters"]
                if trackingParameters is None
                else trackingParameters
            )
            # bdloModel = self.generateModel(bdloModelParameters)
            kinematicsModel.skel.setPositions(bdloModel.getGeneralizedCoordinates())
            # resetModel
            reg = KinematicRegularizedCoherentPointDrift(
                Y=Y,
                model=kinematicsModel,
                qInit=qInit,
                **trackingParameters,
            )
            if visualize:
                if visualizationCallback is None:
                    visualizationCallback = self.getVisualizationCallback(reg)
                reg.registerCallback(visualizationCallback)
        elif method == "krcpd4BDLO":
            trackingParameters = (
                self.config["krcpd4BDLOParameters"]
                if trackingParameters is None
                else trackingParameters
            )
            # bdloModel = self.generateModel(bdloModelParameters)
            kinematicsModel.skel.setPositions(bdloModel.getGeneralizedCoordinates())
            # resetModel
            branchCorrespondances = []
            for branch in bdloModel.getBranches():
                branchCorrespondances.append(
                    branch.getBranchInfo()["correspondingBodyNodeIndices"]
                )
            reg = KinematicRegularizedCoherentPointDrift4BDLO(
                Y=Y,
                model=kinematicsModel,
                qInit=qInit,
                B=branchCorrespondances,
                **trackingParameters,
            )
            if visualize:
                if visualizationCallback is None:
                    visualizationCallback = self.getVisualizationCallback(reg)
                reg.registerCallback(visualizationCallback)
        registrationResult = self.runRegistration(
            reg, checkConvergence=checkConvergence, logTargets=logTargets
        )
        trackingResult["registrations"].append(registrationResult)
        for frame in framesToTrack:
            pointCloud = self.getPointCloud(
                frame,
                dataSetPath,
            )
            reg.setTargetPointCloud(Y=pointCloud[0])
            registrationResult = self.runRegistration(reg)
            trackingResult["registrations"].append(registrationResult)
        return trackingResult

    # -------------------------------------------------------------------------
    # VISUALIZATION FUNCTIONS
    # -------------------------------------------------------------------------
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
            ax.set_zlabel("Z")
        elif ax is None and dim <= 2:
            ax = fig.add_subplot()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
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
        elif (type(classHandle) == CoherentPointDrift) or (
            type(classHandle) == StructurePreservedRegistration
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
        elif type(classHandle) == KinematicsPreservingRegistration:
            plotPointSets(
                ax=ax,
                X=classHandle.T,
                Y=classHandle.Y,
                ySize=1,
                xSize=30,
                xColor=[1, 0, 0],
                yColor=[0, 0, 0],
            )
            plotPointSet(
                ax=ax,
                X=classHandle.X_desired,
                size=30,
                color=[0, 1, 0],
            )
        elif (type(classHandle) == KinematicRegularizedCoherentPointDrift) or (
            type(classHandle) == KinematicRegularizedCoherentPointDrift4BDLO
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
            plotPointSet(ax=ax, X=classHandle.Xreg, size=30, color=[1, 0, 0], alpha=0.1)
        elif type(classHandle) == MinimalSpanningTreeTopology:
            if "color" in kwargs:
                color = kwargs["color"]
            else:
                color = [1, 0, 0]
            pointPairs = classHandle.getAdjacentPointPairs()
            for pointPair in pointPairs:
                stackedPair = np.stack(pointPair)
                plotLine(ax, pointPair=stackedPair, color=color)
        elif type(classHandle) == BranchedDeformableLinearObject:
            if "color" in kwargs:
                color = kwargs["color"]
            else:
                color = [0, 0, 1]
            pointPairs = classHandle.getAdjacentPointPairs()
            for pointPair in pointPairs:
                stackedPair = np.stack(pointPair)
                plotLine(ax, pointPair=stackedPair, color=color)

        elif type(classHandle) == BDLOLocalization:
            templateTopologyColor = list(self.colorMaps["viridis"].to_rgba(0)[:3])
            extractedTopologyColor = list(self.colorMaps["viridis"].to_rgba(1)[:3])
            correspondanceColor = list(self.colorMaps["viridis"].to_rgba(0.5)[:3])
            self.standardVisualizationFunction(
                fig,
                ax,
                classHandle.templateTopology,
                **{"color": templateTopologyColor},
            )
            self.standardVisualizationFunction(
                fig,
                ax,
                classHandle.extractedTopology,
                **{"color": extractedTopologyColor},
            )
            for i, x in enumerate(classHandle.X):
                plotLine(
                    ax=ax,
                    pointPair=np.vstack(
                        ((classHandle.C.T @ classHandle.YTarget)[i], x)
                    ),
                    color=correspondanceColor,
                    alpha=0.3,
                )
        else:
            raise NotImplementedError
        set_axes_equal(ax)

    def visualizeTopologyExtractionResult(self, topologyExtraction, *args, **kwargs):
        fig, ax = self.setupFigure()
        self.standardVisualizationFunction(
            fig, ax, topologyExtraction.extractedTopology
        )
        leafNodeIndices = topologyExtraction.extractedTopology.getLeafNodeIndices()
        branchNodeIndices = topologyExtraction.extractedTopology.getBranchNodeIndices()
        plotPointSet(
            ax=ax,
            X=topologyExtraction.extractedTopology.X,
            color=[0, 0, 1],
            size=30,
            alpha=0.4,
        )
        plotPointSet(
            ax=ax,
            X=topologyExtraction.extractedTopology.X[leafNodeIndices, :],
            color=[1, 0, 0],
            size=50,
            alpha=1,
        )
        plotPointSet(ax=ax, X=topologyExtraction.Y, color=[0, 0, 0], size=1, alpha=0.1)

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

    def visualizeReprojectionError(
        self,
        fileName,
        dataSetPath,
        modelParameters,
        q,
        markerBranchCoordinates,
        groundTruthLabelCoordinates,
        modelColor=[0, 81 / 255, 158 / 255],
        modelThickness=5,
        reprojectionErrorColor=[1, 0, 0],
        reprojectionErrorThickness=5,
        markerThickness=10,
        markerColor=[0, 190 / 255, 1],
        markerFill=-1,
        groundTruthLabelThickness=10,
        groundTruthLabelColor=[255 / 255, 109 / 255, 106 / 255],
        groundTruthLabelFill=-1,
        imageWitdthInInches=5,
        imageHeightInInches=5,
        plotGrayScale=False,
        block=False,
        save=False,
        savePath="data/eval/imgs/",
        format="png",
        dpi=100,
    ):
        """visualizes the 2D reprojection error between estimated marker coordinates of a configuration and annotated ground truth labels

        Args:
            fileName (str): file name
            dataSetPath (str): path to data set
            modelParameters (dict): parameters of the model
            q (np.array): Nx1 array of generalized coordinates for the N degees of freedom
            markerBranchCoordinates (list of tuples): branch local coordinates as tuples of (branchIndex, s)
            groundTruthLabelCoordinates (np.array): Lx2 of pixel coordinates for the L ground truth labels

        """
        # load image
        rgbImg = self.getDataSet(fileName, dataSetPath)[0]  # load image

        # generate model
        model = self.generateModel(modelParameters)
        # get 3D joint coordinates
        jointPositions3D = np.concatenate(
            model.getAdjacentPointPairs(q=q),
            axis=0,
        )
        # get 3D marker coordinates
        markerCoordinates3D = model.computeForwardKinematicsFromBranchLocalCoordinates(
            q=q,
            branchLocalCoordinates=markerBranchCoordinates,
        )
        # reproject markers in 2D pixel coordinates
        markerCoordinates2D = self.reprojectFrom3DRobotBase(
            markerCoordinates3D, dataSetPath
        )
        # reproject joints in 2D pixel coordinates
        jointPositions2D = self.reprojectFrom3DRobotBase(jointPositions3D, dataSetPath)

        # plot image
        # scale colors
        modelColor = tuple([x * 255 for x in modelColor])
        reprojectionErrorColor = tuple([x * 255 for x in reprojectionErrorColor])
        markerColor = tuple([x * 255 for x in markerColor])
        groundTruthLabelColor = tuple([x * 255 for x in groundTruthLabelColor])

        # draw image
        i = 0
        while i <= len(jointPositions2D[:, 0]) - 1:
            cv2.line(
                rgbImg,
                (jointPositions2D[:, 0][i], jointPositions2D[:, 1][i]),
                (jointPositions2D[:, 0][i + 1], jointPositions2D[:, 1][i + 1]),
                modelColor,
                modelThickness,
            )
            i += 2
        for i, markerPosition in enumerate(markerCoordinates2D):
            cv2.line(
                rgbImg,
                (markerPosition[0], markerPosition[1]),
                (
                    groundTruthLabelCoordinates[i][0],
                    groundTruthLabelCoordinates[i][1],
                ),
                reprojectionErrorColor,
                reprojectionErrorThickness,
            )
            cv2.circle(rgbImg, markerPosition, markerThickness, markerColor, markerFill)
            cv2.circle(
                rgbImg,
                groundTruthLabelCoordinates[i],
                groundTruthLabelThickness,
                groundTruthLabelColor,
                groundTruthLabelFill,
            )

        # use matplotlib
        fig = plt.figure(frameon=False)
        fig.set_size_inches(imageWitdthInInches, imageHeightInInches)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        if plotGrayScale:
            ax.imshow(
                cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY), cmap="gray", aspect="auto"
            )
        else:
            ax.imshow(rgbImg, cmap="gray", aspect="auto")
        if save:
            plt.savefig(
                savePath + fileName.split(".")[0] + "_reprojectionError" + "." + format,
                format=format,
                dpi=dpi,
            )
        plt.show(block=block)

    def plotTimeSeries(self, timeSeriesDataY, timeSeriesDataX=None, block=False):
        """Plots a time series of the given data

        Args:
            timeSeriesData (_type_): _description_
        """
        fig, ax = self.setupFigure(dim=2)
        numTimeSeries = len(timeSeriesDataY)
        for i, Y in enumerate(timeSeriesDataY):
            if timeSeriesDataX is None:
                X = np.array(list(range(len(Y))))
            if numTimeSeries > 1:
                color = self.colorMaps["viridis"].to_rgba(i / (numTimeSeries - 1))
            else:
                color = [0, 0, 1]
            ax.plot(X, Y, color=color)
        plt.show(block=block)

    def getModelID(self, modelName: str):
        mappingDict = {
            "singleDLO": 0,
            "modelY": 1,
            "partialWireHarness": 2,
            "arenaWireHarness": 3,
        }
        for entry in mappingDict:
            if entry == modelName:
                return mappingDict[entry]
        return None
