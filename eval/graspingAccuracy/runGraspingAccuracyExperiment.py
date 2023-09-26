import sys
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for graspingAccuracy evaluation script failed.")
    raise

runOpt = {
    "dataSetsToEvaluate": [-1],
    "runInitialLocalization": False,
    "runTracking": True,
    "localizationOptions": {
        "initializationFrame": 0,
        "run2DSkeletonization": False,
        "filterLOF": True,
        "runL1": True,
        "runTopologyExtraction": True,
        "runLocalization": True,
    },
    "trackingOptions": {
        "registrationMethods": [
            "cpd",
            "spr",
            "kpr",
            "krcpd",
        ],  # "cpd", "spr", "kpr", "krcpd"
        "modelBasedMethods": ["kpr", "krcpd"],
    },
    "verbose": True,
}
saveOpt = {
    "saveLocalizationResults": False,
    "saveTrackingResults": True,
    "resultFolderPath": "data/eval/graspingAccuracy/results",
    "resultFileType": ".pkl",
}
visOpt = {
    "visPointCloud": True,
    "visLOFFilter": True,
    "visL1MedianIterations": True,
    "visL1MedianResult": True,
    "visTopologyExtractionResult": True,
    "visLocalizationIterations": True,
    "visLocalizationResult": True,
    "visTrackingIterations": True,
}
dataSetPaths = [
    # "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_130903_modelY/",
    # "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_131545_modelY/",
    # "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_154903_modelY/",
    # "data/darus_data_download/data/20230807_RoboticWireHarnessMounting/20230807_142319_partial/",
    # "data/darus_data_download/data/20230807_RoboticWireHarnessMounting/20230807_142909_partial/",
    # "data/darus_data_download/data/20230807_RoboticWireHarnessMounting/20230807_143737_partial/",
    "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/",
    "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/",
    "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_142058_arena/",
]

if __name__ == "__main__":
    print("Starting evaluation")
    if runOpt["dataSetsToEvaluate"][0] == -1:
        dataSetsToEvaluate = dataSetPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(dataSetPaths)
            if i in runOpt["dataSetsToEvaluate"]
        ]
    for dataSetPath in dataSetsToEvaluate:
        # setup evaluation
        dataSetName = dataSetPath.split("/")[-2]
        dataSetIdentifier = dataSetName
        relConfigFilePath = configFilePath = (
            "/evalConfigs/evalConfig" + "_" + dataSetIdentifier + ".json"
        )
        pathToConfigFile = (
            os.path.dirname(os.path.abspath(__file__)) + relConfigFilePath
        )
        eval = GraspingAccuracyEvaluation(pathToConfigFile)
        dataSetName = dataSetPath.split("/")[-2]
        model, modelParameters = eval.getModel(dataSetPath)
        modelIdentifier = dataSetName.split("_")[-1]

        # setup result
        resultFileName = "result"
        resultFolderPath = os.path.join(saveOpt["resultFolderPath"], dataSetName)
        # create directory if it does not exist
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath, exist_ok=True)
        resultFilePath = (
            os.path.join(resultFolderPath, resultFileName) + saveOpt["resultFileType"]
        )
        result = {}
        result["config"] = eval.config
        result["dataSetPath"] = dataSetPath
        result["dataSetName"] = dataSetName
        result["modelIdentifier"] = modelIdentifier
        result["modelParameters"] = modelParameters

        # run initialization
        if runOpt["runInitialLocalization"]:
            frame = runOpt["localizationOptions"]["initializationFrame"]
            fileName = eval.getFileName(frame, dataSetPath)
            filePath = eval.getFilePath(frame, dataSetPath)
            result["initializationResult"] = {
                "fileName": fileName,
                "filePath": filePath,
                "frame": frame,
            }
            # load data
            if runOpt["localizationOptions"]["run2DSkeletonization"]:
                pointCloud = eval.getPointCloud(
                    frame, dataSetPath, segmentationMethod="skeletonized"
                )
            else:
                pointCloud = eval.getPointCloud(frame, dataSetPath)
            result["initializationResult"]["pointCloud"] = pointCloud
            Y = pointCloud[0]
            if visOpt["visPointCloud"]:
                fig, ax = setupLatexPlot3D()
                plotPointSet(ax=ax, X=Y, color=[1, 0, 0], size=5, alpha=0.1)
                ax.view_init(elev=50, azim=40)
                plt.show(block=True)

            # filter outliers with lof filter
            if runOpt["localizationOptions"]["filterLOF"]:
                lofResult = eval.filterLOF(Y)
                result["initializationResult"]["lofResult"] = lofResult
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
            if runOpt["localizationOptions"]["runL1"]:
                l1Result = eval.runL1Median(
                    pointSet=Y,
                    visualizeIterations=visOpt["visL1MedianIterations"],
                    visualizeResult=False,
                    block=False,
                    closeAfterRunning=True,
                )
                Y = l1Result["T"]
                # save result
                result["initializationResult"]["l1Result"] = l1Result

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

            # RODO: add SOM

            if runOpt["localizationOptions"]["runTopologyExtraction"]:
                # extract minimum spanning tree
                topologyExtractionResult = eval.extractMinimumSpanningTreeTopology(
                    Y, model
                )
                extractedTopology = topologyExtractionResult["extractedTopology"]
                # save result
                result["initializationResult"][
                    "topologyExtractionResult"
                ] = topologyExtractionResult
                if visOpt["visTopologyExtractionResult"]:
                    # visualize result
                    fig, ax = setupLatexPlot3D()
                    plotGraph3D(
                        ax=ax,
                        X=extractedTopology.X,
                        adjacencyMatrix=extractedTopology.featureMatrix,
                    )
                    plt.show(block=True)
            if runOpt["localizationOptions"]["runLocalization"]:
                localizationResult, _ = eval.runInitialLocalization(
                    pointSet=Y,
                    extractedTopology=extractedTopology,
                    bdloModelParameters=modelParameters,
                    visualizeCorresponanceEstimation=True,
                    visualizeIterations=visOpt["visLocalizationIterations"],
                    visualizeResult=False,
                    block=False,
                    closeAfterRunning=True,
                    logResults=True,
                )
                result["initializationResult"][
                    "localizationResult"
                ] = localizationResult

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
            # save initializaiton result
            if saveOpt["saveLocalizationResults"]:
                # check if result already exist
                if os.path.exists(resultFilePath):
                    existingResults = eval.loadResults(resultFilePath)
                    # add exisiting tracking results
                    if "trackingResults" in existingResults:
                        result["trackingResults"] = existingResults["trackingResults"]
                eval.saveWithPickle(
                    data=result,
                    filePath=resultFilePath,
                    recursionLimit=10000,
                    verbose=True,
                )
        else:
            if os.path.exists(resultFilePath):
                result = eval.loadResults(resultFilePath)
            else:
                raise ValueError(
                    "No result exists yet. Run localization first to proceede."
                )

        if runOpt["runTracking"]:
            # setup tracking problem
            framesToTrack = list(range(0, eval.getNumImageSetsInDataSet(dataSetPath)))[
                ::3
            ]
            methodsToEvaluate = runOpt["trackingOptions"]["registrationMethods"]
            modelBasedMethods = runOpt["trackingOptions"]["modelBasedMethods"]
            result["trackingResults"] = {}
            for method in methodsToEvaluate:
                result["trackingResults"][method] = {
                    "method": method,
                    "modelParameters": modelParameters,
                    "registrationResults": [],
                    "B": result["initializationResult"]["localizationResult"]["BInit"],
                    "S": result["initializationResult"]["localizationResult"]["S"],
                    "adjacencyMatrix": model.getBodyNodeNodeAdjacencyMatrix(),
                }
                for numFrame, frame in enumerate(framesToTrack):
                    # load point cloud data
                    Y, colors = eval.getPointCloud(frame, dataSetPath)

                    # setup result
                    registrationResult = {
                        "frame": frame,
                        "fileName": eval.getFileName(frame, dataSetPath),
                        "filePath": eval.getFilePath(frame, dataSetPath),
                    }
                    # setup registration
                    registrationConfig = {}
                    registrationConfig["Y"] = Y
                    if numFrame == 0 and method in modelBasedMethods:
                        qInit = result["initializationResult"]["localizationResult"][
                            "qInit"
                        ]
                        registrationConfig["qInit"] = qInit
                    elif numFrame == 0 and method not in modelBasedMethods:
                        XInit = result["initializationResult"]["localizationResult"][
                            "XInit"
                        ]
                        registrationConfig["X"] = XInit
                    elif numFrame != 0 and method in modelBasedMethods:
                        qInit = result["trackingResults"][method][
                            "registrationResults"
                        ][-1]["result"]["q"]
                        registrationConfig["qInit"] = qInit

                        (b, s) = eval.loadGraspingLocalCoordinates(dataSetPath)[
                            numFrame - 1
                        ]
                        graspedBodyNode = (
                            model.getBodyNodeIndexFromBranchLocalCoodinate(b, s)
                        )
                        releasePosition = eval.loadGroundTruthGraspingPose(
                            dataSetPath, frame - 1
                        )[1]
                        registrationConfig["constraints"] = {}
                        registrationConfig["constraints"]["constrainedNodeIndices"] = [
                            graspedBodyNode
                        ]
                        registrationConfig["constraints"]["constrainedPositions"] = [
                            releasePosition
                        ]
                    else:
                        XInit = result["trackingResults"][method][
                            "registrationResults"
                        ][-1]["result"]["T"]
                        registrationConfig["X"] = XInit

                    registrationConfig["model"] = KinematicsModelDart(
                        model.skel.clone()
                    )
                    registration = eval.setupRegistration(
                        method,
                        registrationConfig,
                        visualizeIterations=visOpt["visTrackingIterations"],
                    )
                    registration.register(checkConvergence=False)
                    plt.close("all")
                    registrationResult["result"] = registration.getResults()
                    # save registration result
                    result["trackingResults"][method]["registrationResults"].append(
                        registrationResult
                    )
                # save tracking result for frame
                if saveOpt["saveTrackingResults"]:
                    # check if result already exist
                    if os.path.exists(resultFilePath):
                        existingResults = eval.loadResults(resultFilePath)
                        if "trackingResults" in existingResults:
                            existingTrackingResults = eval.loadResults(resultFilePath)[
                                "trackingResults"
                            ]
                            # add exisiting tracking results to result
                            for existingMethod in existingTrackingResults:
                                if existingMethod != method:
                                    result["trackingResults"][
                                        existingMethod
                                    ] = existingTrackingResults[existingMethod]
                    eval.saveWithPickle(
                        data=result,
                        filePath=resultFilePath,
                        recursionLimit=10000,
                        verbose=True,
                    )
    if runOpt["verbose"]:
        print("Finished evaluation.")
