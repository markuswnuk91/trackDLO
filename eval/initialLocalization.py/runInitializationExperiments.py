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
    from src.localization.downsampling.random.randomSampling import RandomSampling
except:
    print("Imports for initialization evaluation script failed.")
    raise

runOpt = {
    "dataSetsToEvaluate": [0],
    "framesToEvaluate": [-1],
    "runInitializationExperiment": True,
    # "runSom": False,
    "run2DSkeletonization": True,
    "runL1": True,
    "filterLOF": True,
    "runTopologyExtraction": True,
    "runLocalization": True,
    #    "evaluation": True
}
saveOpt = {
    "saveResults": True,
    "resultRootFolderPath": "data/eval/initialLocalization/results",
    "resultFileType": ".pkl",
    "overwrite": True,
}
visOpt = {
    "visPointCloud": True,
    "visLOFFilter": True,
    "visL1MedianIterations": True,
    "visL1MedianResult": False,
    # "visSOMIterations": True,
    # "visSOMResult": True,
    "visTopologyExtractionResult": True,
    "visLocalizationIterations": True,
    "visLocalizationResult": False,
}
# framesToSkip = [0, 1, 2, 5, 6, 7, 8, 9, 14, 15, 16, 23, 24, 34, 35, 44, 45, 47]
framesToSkip = []
failedFrames = []

dataSetPaths = [
    # "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/",
    # "data/darus_data_download/data/20230807_Configurations_mounted/20230807_150735_partial/",
    # "data/darus_data_download/data/202230603_Configurations_mounted/20230603_140143_arena/",
    # "data/darus_data_download/data/20230516_Configurations_labeled/20230516_112207_YShape/",
    # "data/darus_data_download/data/20230516_Configurations_labeled/20230516_113957_Partial/",  # finished 10.09.2023
    # "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/", # finished 12.09.2023
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
            dataSetIdentifier = dataSetName
            relConfigFilePath = configFilePath = (
                "/evalConfigs/evalConfig" + "_" + dataSetIdentifier + ".json"
            )
            modelIdentifier = dataSetName.split("_")[-1]
            pathToConfigFile = (
                os.path.dirname(os.path.abspath(__file__)) + relConfigFilePath
            )
            eval = InitialLocalizationEvaluation(pathToConfigFile)
            model, modelParameters = eval.getModel(dataSetPath)
            resultFolderPath = os.path.join(
                saveOpt["resultRootFolderPath"], dataSetName
            )
            # create directory if it does not exist
            if not os.path.exists(resultFolderPath):
                os.makedirs(resultFolderPath, exist_ok=True)
            # configure logging
            logFilePath = os.path.join(resultFolderPath, "localization.log")
            logging.basicConfig(filename=logFilePath, level=logLevel, format=logFormat)

            framesToEvaluate = runOpt["framesToEvaluate"]
            if framesToEvaluate[0] == -1:
                numFramesInDataSet = eval.getNumImageSetsInDataSet(dataSetPath)
                framesToEvaluate = list(range(0, numFramesInDataSet))
            framesToEvaluate = [
                frame
                for frame in framesToEvaluate
                if (
                    frame not in eval.config["invalidFrames"]
                    and frame not in framesToSkip
                )
            ]
            for i, frame in enumerate(framesToEvaluate):
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
                    if os.path.exists(resultFilePath):
                        result = eval.loadResults(resultFilePath)
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
                    if "run2DSkeletonization":
                        pointCloud = eval.getPointCloud(
                            frame, dataSetPath, segmentationMethod="skeletonized"
                        )
                    else:
                        pointCloud = eval.getPointCloud(frame, dataSetPath)
                    Y = pointCloud[0]
                    if visOpt["visPointCloud"]:
                        fig, ax = setupLatexPlot3D()
                        plotPointSet(ax=ax, X=Y, color=[1, 0, 0], size=5, alpha=0.1)
                        ax.view_init(elev=50, azim=40)
                        plt.show(block=True)

                    # filter outliers with lof filter
                    if runOpt["filterLOF"]:
                        lofResult = eval.filterLOF(Y)
                        result["lofResult"] = lofResult
                        Y = lofResult["filteredPointSet"]
                        if saveOpt["saveResults"]:
                            eval.saveWithPickle(
                                data=result,
                                filePath=resultFilePath,
                                recursionLimit=10000,
                            )
                    else:
                        if "lofResult" in result:
                            lofResult = result["lofResult"]
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
                        ax.view_init(elev=50, azim=40)
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
                        if saveOpt["saveResults"]:
                            eval.saveWithPickle(
                                data=result,
                                filePath=resultFilePath,
                                recursionLimit=10000,
                            )
                    elif not runOpt["runL1"] and "l1Result" in result:
                        l1Result = result["l1Result"]
                        Y_hat = l1Result["T"]
                        # rand = RandomSampling(
                        #     eval.config["topologyExtraction"]["l1Parameters"][
                        #         "numSeedPoints"
                        #     ]
                        # )
                        # Y_hat = rand.sampleRandom(Y)
                    else:
                        pass

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

                    # # perform som
                    # if runOpt["runSom"]:
                    #     somResult = eval.runSOM(
                    #         pointSet=Y,
                    #         visualizeIterations=visOpt["visSOMIterations"],
                    #         visualizeResult=visOpt["visSOMResult"],
                    #         block=True,
                    #         closeAfterRunning=True,
                    #     )
                    #     Y_hat = somResult["T"]
                    #     # save result
                    #     result["somResult"] = somResult

                    if runOpt["runTopologyExtraction"]:
                        # extract minimum spanning tree
                        topologyExtractionResult = (
                            eval.extractMinimumSpanningTreeTopology(Y_hat, model)
                        )
                        extractedTopology = topologyExtractionResult[
                            "extractedTopology"
                        ]
                        # save result
                        result["topologyExtractionResult"] = topologyExtractionResult
                        if saveOpt["saveResults"]:
                            eval.saveWithPickle(
                                data=result,
                                filePath=resultFilePath,
                                recursionLimit=10000,
                            )
                    elif (
                        not runOpt["runTopologyExtraction"]
                        and "topologyExtractionResult" in result
                    ):
                        topologyExtractionResult = result["topologyExtractionResult"]
                        extractedTopology = topologyExtractionResult[
                            "extractedTopology"
                        ]
                    else:
                        raise ValueError(
                            "Cannot continue because no topology was extracted. Check the settings."
                        )
                    if visOpt["visTopologyExtractionResult"]:
                        # visualize result
                        fig, ax = setupLatexPlot3D()
                        plotGraph(
                            ax=ax,
                            X=extractedTopology.X,
                            adjacencyMatrix=extractedTopology.featureMatrix,
                        )
                        plt.show(block=True)

                    if runOpt["runLocalization"]:
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
                        result["localizationResult"] = localizationResult
                        if saveOpt["saveResults"]:
                            eval.saveWithPickle(
                                data=result,
                                filePath=resultFilePath,
                                recursionLimit=10000,
                                verbose=True,
                            )
                    elif (
                        not runOpt["runLocalization"] and "localizationResult" in result
                    ):
                        localizationResult = result["localizationResult"]
                    else:
                        raise ValueError(
                            "Cannot continue because localization was not performed. Check the settings."
                        )
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
                    print(
                        "Failed on file{} in data set {}".format(
                            failedFileName, dataSetPath
                        )
                    )
                    traceback.print_exc()
                print(
                    "Evaluated frame {} ({}/{})".format(
                        frame, i + 1, len(framesToEvaluate)
                    )
                )
            print("Finished experiments on data set: {}".format(dataSetName))
            logging.info("Finished experiments on data set: {}".format(dataSetName))
        print("Finished experiments.")
