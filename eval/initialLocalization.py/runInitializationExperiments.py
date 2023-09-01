import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import johnson

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

framesToEvaluate = [0, 1, 2, 3, 4, 5, 6]

dataSetPaths = [
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/",
    "data/darus_data_download/data/20230807_Configurations_mounted/20230807_150735_partial/",
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_140143_arena/",
]

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
            for frame in framesToEvaluate:
                initializationResult = {}
                # load data
                pointCloud = eval.getPointCloud(frame, dataSetPath)
                Y = pointCloud[0]

                # fig, ax = setupLatexPlot3D()
                # plotPointSet(ax=ax, X=Y, color=[1, 0, 0])
                if runOpt["filterLOF"]:
                    Y = eval.filterLOF(Y)
                # plotPointSet(ax=ax, X=Y, color=[0, 0, 1])
                # plt.show(block=True)
                # perform l1 skeletonization
                if runOpt["runL1"]:
                    l1Result = eval.runL1Median(
                        pointSet=Y,
                        visualizeIterations=True,
                        visualizeResult=True,
                        block=False,
                        closeAfterRunning=True,
                    )
                    Y_hat = l1Result["T"]
                # save result

                # perform som
                if runOpt["runSom"]:
                    somResult = eval.runSOM(
                        pointSet=Y,
                        visualizeIterations=True,
                        visualizeResult=True,
                        block=True,
                        closeAfterRunning=True,
                    )
                    Y_hat = somResult["T"]
                # save result

                # extract minimum spanning tree
                nLongestPaths = model.getNumLeafNodes() - 1
                extractedTopology = eval.extractMinimumSpanningTreeTopology(
                    Y_hat, nLongestPaths
                )
                # visualize result
                fig, ax = setupLatexPlot3D()
                plotGraph(
                    ax=ax,
                    X=extractedTopology.X,
                    adjacencyMatrix=extractedTopology.featureMatrix,
                )
                plt.show(block=True)
                # save result

                eval.runInitialLocalization(
                    pointSet=Y_hat,
                    extractedTopology=extractedTopology,
                    bdloModelParameters=modelParameters,
                    visualizeCorresponanceEstimation=True,
                    visualizeIterations=True,
                    visualizeResult=True,
                    block=True,
                    closeAfterRunning=True,
                    logResults=True,
                )
                # perform correspondance assignment
                # visualize result
                # save result

                # perform inverse kinematics
                # visualize result
                # save result

                # perform tracking
                # visualize result
                # save result
