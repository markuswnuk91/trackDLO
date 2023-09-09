import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import johnson
import logging
import traceback


try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )

    # visualization
    from src.visualization.plot3D import *

    from src.utils.utils import minimalSpanningTree
    from scipy.spatial import distance_matrix
except:
    print("Imports for initialization evaluation script failed.")
    raise

runOpt = {
    "dataSetsToEvaluate": [1],
    "runInitializationExperiment": True,
    "runSom": False,
    "runL1": True,
    "filterLOF": True,
    #    "evaluation": True
}
saveOpt = {
    "saveResults": True,
    "resultRootFolderPath": "data/eval/initialLocalization/results",
    "resultFileType": ".pkl",
    "overwrite": True,
}
visOpt = {
    "visPointCloud": False,
    "visLOFFilter": False,
    "visL1MedianIterations": True,
    "visL1MedianResult": False,
    "visSOMIterations": False,
    "visSOMResult": False,
    "visTopologyExtractionResult": False,
    "visLocalizationIterations": True,
    "visLocalizationResult": False,
}
framesToEvaluate = [-1]
framesToSkip = []
failedFrames = []

dataSetPaths = [
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/",
    "data/darus_data_download/data/20230807_Configurations_mounted/20230807_150735_partial/",
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_140143_arena/",
]
# logging settings
logLevel = logging.INFO
logFormat = "%(asctime)s - %(levelname)s - %(message)s"

if __name__ == "__main__":
    if runOpt["dataSetsToEvaluate"][0] == -1:
        dataSetsToEvaluate = dataSetPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(dataSetPaths)
            if i in runOpt["dataSetsToEvaluate"]
        ]

    # determine data sets without too much occlusion/noise

    # run initialization experiments
    if runOpt["runInitializationExperiment"]:
        failCounter = 0
        for dataSetPath in dataSetsToEvaluate:
            # setup evaluation class
            dataSetName = dataSetPath.split("/")[-2]
            modelIdentifier = dataSetName.split("_")[-1]
            relConfigFilePath = configFilePath = (
                "/evalConfigs/evalConfig" + "_" + modelIdentifier + ".json"
            )
            pathToConfigFile = (
                os.path.dirname(os.path.abspath(__file__)) + relConfigFilePath
            )
            eval = InitialLocalizationEvaluation(pathToConfigFile)
            model, modelParameters = eval.getModel(dataSetPath)
            resultFolderPath = os.path.join(
                saveOpt["resultRootFolderPath"], dataSetName
            )
            # configure logging
            logFilePath = os.path.join(resultFolderPath, "localization.log")
            logging.basicConfig(filename=logFilePath, level=logLevel, format=logFormat)

            if framesToEvaluate[0] == -1:
                numFramesInDataSet = eval.getNumImageSetsInDataSet(dataSetPath)
                framesToEvaluate = list(range(0, numFramesInDataSet))
            framesToEvaluate = [
                frame for frame in framesToEvaluate if frame not in framesToSkip
            ]
            for frame in framesToEvaluate:
                try:
                    # setup result file
                    fileName = eval.getFileName(frame, dataSetFolderPath=dataSetPath)
                    filePath = eval.getFilePath(frame, dataSetPath)
                    resultFileName = "_".join(fileName.split("_")[0:3]) + "_result"
                    resultFilePath = (
                        os.path.join(resultFolderPath, resultFileName)
                        + saveOpt["resultFileType"]
                    )
                    # check if result for this frame exists, if not create new result
                    if os.path.exists(resultFilePath) and not saveOpt["overwrite"]:
                        eval.loadResults(resultFilePath)
                    else:
                        result = {}

                    result["config"] = eval.config
                    result["dataSetPath"] = dataSetPath
                    result["dataSetName"] = dataSetName
                    result["fileName"] = fileName
                    result["filePath"] = filePath
                    result["frame"] = frame
                    result["modelIdentifier"] = modelIdentifier
                    result["modelParameters"] = modelParameters

                    # load data
                    pointCloud = eval.getPointCloud(frame, dataSetPath)
                    Y = pointCloud[0]
                    result["pointCloud"] = Y
                    if visOpt["visPointCloud"]:
                        fig, ax = setupLatexPlot3D()
                        plotPointSet(ax=ax, X=Y, color=[1, 0, 0], size=5, alpha=0.1)
                        plt.show(block=True)

                    # filter outliers with lof filter
                    if runOpt["filterLOF"]:
                        lofResult = eval.filterLOF(Y)
                    result["lofResult"] = lofResult
                    Y = lofResult["filteredPointSet"]
                    if visOpt["visLOFFilter"]:
                        fig, ax = setupLatexPlot3D()
                        plotPointSets(
                            ax=ax,
                            X=Y,
                            Y=lofResult["outliers"],
                            xColor=[0, 0, 0],
                            yColor=[1, 0, 0],
                            xSize=5,
                            ySize=30,
                        )
                        plt.show(block=True)

                    # perform l1 skeletonization
                    if runOpt["runL1"]:
                        l1Result = eval.runL1Median(
                            pointSet=Y,
                            visualizeIterations=visOpt["visL1MedianIterations"],
                            visualizeResult=False,
                            block=False,
                            closeAfterRunning=True,
                        )
                        Y_hat = l1Result["T"]
                        # save result
                        result["l1Result"] = l1Result
                    if visOpt["visL1MedianResult"]:
                        fig, ax = setupLatexPlot3D()
                        plotPointSets(
                            ax=ax,
                            X=Y,
                            Y=l1Result["T"],
                            xColor=[0, 0, 0],
                            yColor=[1, 0, 0],
                            xSize=5,
                            ySize=30,
                        )
                        plt.show(block=True)

                    # perform som
                    if runOpt["runSom"]:
                        somResult = eval.runSOM(
                            pointSet=Y,
                            visualizeIterations=visOpt["visSOMIterations"],
                            visualizeResult=visOpt["visSOMResult"],
                            block=True,
                            closeAfterRunning=True,
                        )
                        Y_hat = somResult["T"]
                        # save result
                        result["somResult"] = somResult

                    # extract minimum spanning tree
                    topologyExtractionResult = eval.extractMinimumSpanningTreeTopology(
                        Y_hat, model
                    )
                    extractedTopology = topologyExtractionResult["extractedTopology"]
                    # save result
                    result["topologyExtractionResult"] = topologyExtractionResult
                    if visOpt["visTopologyExtractionResult"]:
                        # visualize result
                        fig, ax = setupLatexPlot3D()
                        plotGraph(
                            ax=ax,
                            X=extractedTopology.X,
                            adjacencyMatrix=extractedTopology.featureMatrix,
                        )
                        plt.show(block=True)

                    localizationResult, _ = eval.runInitialLocalization(
                        pointSet=Y_hat,
                        extractedTopology=extractedTopology,
                        bdloModelParameters=modelParameters,
                        visualizeCorresponanceEstimation=True,
                        visualizeIterations=visOpt["visLocalizationIterations"],
                        visualizeResult=False,
                        block=False,
                        closeAfterRunning=True,
                        logResults=True,
                    )
                    # save result
                    result["localizationResult"] = localizationResult
                    if visOpt["visLocalizationResult"]:
                        fig, ax = setupLatexPlot3D()
                        plotTopology3D(
                            ax=ax,
                            topology=localizationResult["extractedTopology"],
                            color=[1, 0, 0],
                            lineAlpha=1,
                        )
                        model.setGeneralizedCoordinates(localizationResult["q"])
                        plotTopology3D(ax=ax, topology=model)
                        plotCorrespondances3D(
                            ax=ax,
                            X=localizationResult["XCorrespondance"],
                            Y=localizationResult["YTarget"],
                            C=localizationResult["C"],
                            xColor=[0, 0, 1],
                            yColor=[1, 0, 0],
                            correspondanceColor=[0, 0, 0],
                            xSize=20,
                            ySize=20,
                            xAlpha=0.3,
                            yAlpha=0.3,
                            lineAlpha=0.3,
                        )
                        plt.show(block=True)

                    if saveOpt["saveResults"]:
                        eval.saveWithPickle(
                            data=result, filePath=resultFilePath, recursionLimit=10000
                        )
                        # eval.saveResults(
                        #     folderPath=resultFolderPath + "/",
                        #     generateUniqueID=False,
                        #     fileName=resultFileName,
                        #     results=result,
                        #     promtOnSave=False,
                        #     overwrite=True,
                        # )
                    logging.info(
                        'Finished localization for frame {} ({}) from data set: "{}"'.format(
                            frame, fileName, dataSetPath
                        )
                    )
                except:
                    failCounter += 1
                    failedFilePath = eval.getFilePath(frame, dataSetPath)
                    failedFileName = eval.getFileName(frame, dataSetPath)
                    failedFrames.append(failedFilePath)
                    logging.info(
                        'Failed on frame {} ({}) from data set: "{}"'.format(
                            frame, failedFileName, dataSetPath
                        )
                    )
            print("Finished experiments on data set: {}".format(dataSetName))
            logging.info("Finished experiments on data set: {}".format(dataSetName))
        print("Finished experiments.")
