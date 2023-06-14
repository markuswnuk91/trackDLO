import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.evaluation.evaluation import Evaluation

    # tracking algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import KinematicsPreservingRegistration
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.tracking.krspr.krspr import (
        KinematicRegularizedStructurePreservedRegistration,
    )
    from src.tracking.krcpd.krcpd import (
        KinematicRegularizedCoherentPointDrift,
    )

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global vis
global result
vis = True
save = True
loadInitialStateFromResult = False
runExperiment = True
registrationsToRun = ["cpd", "spr", "krcpd"]
# setup evalulation class
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = Evaluation(configFilePath=pathToConfigFile)
# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetToLoad"]]
dataSetName = eval.config["dataSetPaths"][0].split("/")[-2]
resultFolderPath = "data/eval/tracking/" + dataSetName + "/"
resultFileName = "result"
resultFilePath = resultFolderPath + resultFileName + ".pkl"


def setupResultTemplate(dataSetPath):
    # setup result file
    resultTemplate = {
        "dataSetPath": dataSetPath,
        "evalConfig": eval.config,
        "modelGeneration": {},
        "initialization": {
            "pointCloud": None,
            "topologyExtraction": {},
            "localization": {},
        },
        "tracking": {},
    }
    return resultTemplate


def setupTrackingResultTemplate():
    trackingResultTemplate = {
        "method": "",
        "results": [],
    }
    return trackingResultTemplate


def setupRegistrationResultTemplate():
    # setup tracking result
    registrationResultTemplate = {
        "Y": None,
        "X": None,
        "T": [],
        "iteration": [],
        "runtimePerIteration": [],
    }
    return registrationResultTemplate


def runModelGeneration(dataSetPath):
    bdloModel = eval.generateModel(
        dataSetPath, eval.config["modelGeneration"]["numSegments"]
    )
    return bdloModel


def loadPointCloud(dataSetPath, fileNumber):
    pointCloud = eval.getPointCloud(
        fileNumber,
        dataSetPath,
    )
    return pointCloud


def runTopologyExtraction(pointSet):
    # extract topology
    extractedTopology, topologyExtraction = eval.extractTopology(
        pointSet,
        somParameters=eval.config["topologyExtraction"]["somParameters"],
        l1Parameters=eval.config["topologyExtraction"]["l1Parameters"],
        pruningThreshold=eval.config["topologyExtraction"]["pruningThreshold"],
        skeletonize=True,
        visualizeSOMIteration=True,
        visualizeSOMResult=True,
        visualizeL1Iterations=True,
        visualizeL1Result=True,
        visualizeExtractionResult=True,
    )
    # eval.results["initialization"]["topologyExtraction"][
    #     "topologyExtraction"
    # ] = topologyExtraction
    eval.results["initialization"]["topologyExtraction"][
        "extractedTopology"
    ] = extractedTopology
    somResult = {
        "X": topologyExtraction.selfOrganizingMap.X,
        "Y": topologyExtraction.selfOrganizingMap.Y,
        "T": topologyExtraction.selfOrganizingMap.T,
    }
    l1Result = {
        "X": topologyExtraction.l1Median.X,
        "Y": topologyExtraction.l1Median.Y,
        "T": topologyExtraction.l1Median.T,
    }
    eval.results["initialization"]["topologyExtraction"]["som"] = somResult
    eval.results["initialization"]["topologyExtraction"]["l1"] = l1Result
    return extractedTopology


def runInitialLocalization(dataSetPath, initialFrame):
    # load first point cloud of the data set
    pointCloud = loadPointCloud(dataSetPath, initialFrame)
    eval.results["initialization"]["pointCloud"] = pointCloud
    Y = pointCloud[0]
    extractedTopology = runTopologyExtraction(Y)
    # get the model
    # bdloModel = eval.results["modelGeneration"]["model"]
    bdloModel = runModelGeneration(dataSetPath)
    # perform initial localization
    XResult, qResult, localization = eval.initialLocalization(
        pointSet=Y,
        extractedTopology=extractedTopology,
        bdloModel=bdloModel,
        numSamples=eval.config["localization"]["numSamples"],
        numIterations=eval.config["localization"]["numIterations"],
        verbose=eval.config["localization"]["verbose"],
        method=eval.config["localization"]["method"],
        jacobianDamping=eval.config["localization"]["jacobianDamping"],
        visualizeCorresponanceEstimation=True,
        visualizeIterations=True,
        visualizeResult=True,
        visualizationCallback=None,
        block=False,
    )
    # eval.results["initialization"]["localization"]["localization"] = localization
    eval.results["initialization"]["initialFrame"] = initialFrame
    eval.results["initialization"]["localization"]["S"] = localization.S
    eval.results["initialization"]["localization"]["C"] = localization.C
    eval.results["initialization"]["localization"]["XLog"] = localization.XLog
    eval.results["initialization"]["localization"]["qLog"] = localization.qLog
    eval.results["initialization"]["localization"]["XResult"] = XResult
    eval.results["initialization"]["localization"]["qResult"] = qResult
    return


def runExperiment(dataSetPath, startFrame, endFrame):
    # get the model
    # bdloModel = eval.results["modelGeneration"]["model"]
    bdloModel = runModelGeneration(dataSetPath)
    # setup registrations
    XInit = eval.results["initialization"]["localization"]["XResult"]
    qInit = eval.results["initialization"]["localization"]["qResult"]
    Y = eval.results["initialization"]["pointCloud"][0]
    framesToTrack = list(range(startFrame, endFrame, eval.config["frameStep"]))

    # cpd
    if "cpd" in registrationsToRun:
        cpd = CoherentPointDrift(Y=Y, X=XInit, **eval.config["cpdParameters"])
        if vis:
            visualizationCallback_cpd = eval.getVisualizationCallback(cpd)
            cpd.registerCallback(visualizationCallback_cpd)
        # setup result files
        trackingResult = setupTrackingResultTemplate()
        trackingResult["method"] = "cpd"
        eval.results["tracking"]["cpd"] = trackingResult
        for frame in framesToTrack:
            pointCloud = eval.getPointCloud(
                frame,
                dataSetPath,
            )
            Y = pointCloud[0]
            cpd.Y = Y
            cpd.X = cpd.T.copy()
            # setup result callback
            logTargets = lambda: registrationResult["T"].append(cpd.T.copy())
            registrationResult = setupRegistrationResultTemplate()

            # run registration
            cpd.register(checkConvergence=False, customCallback=logTargets)
            # save results
            registrationResult["X"] = cpd.X.copy()
            registrationResult["Y"] = cpd.Y.copy()
            registrationResult["W"] = cpd.W.copy()
            registrationResult["G"] = cpd.G.copy()
            eval.results["tracking"]["cpd"]["results"].append(registrationResult)
    # # spr
    if "spr" in registrationsToRun:
        spr = StructurePreservedRegistration(
            Y=Y, X=XInit, **eval.config["sprParameters"]
        )
        if vis:
            visualizationCallback_spr = eval.getVisualizationCallback(spr)
            spr.registerCallback(visualizationCallback_spr)
        spr.register(checkConvergence=False)
        # setup result files
        trackingResult = setupTrackingResultTemplate()
        trackingResult["method"] = "spr"
        eval.results["tracking"]["spr"] = trackingResult
        for frame in framesToTrack:
            pointCloud = eval.getPointCloud(
                frame,
                dataSetPath,
            )
            Y = pointCloud[0]
            spr.Y = Y

            # setup result callback
            logTargets = lambda: registrationResult["T"].append(spr.T.copy())
            registrationResult = setupRegistrationResultTemplate()

            # run registration
            spr.register(checkConvergence=False, customCallback=logTargets)
            spr.X = spr.T.copy()
            # save results
            registrationResult["X"] = spr.X.copy()
            registrationResult["Y"] = spr.Y.copy()
            registrationResult["W"] = spr.W.copy()
            registrationResult["G"] = spr.G.copy()
            eval.results["tracking"]["spr"]["results"].append(registrationResult)
    # kpr
    if "kpr" in registrationsToRun:
        kpr = KinematicsPreservingRegistration(
            Y=Y,
            qInit=qInit,
            model=KinematicsModelDart(bdloModel.skel.clone()),
            **eval.config["kprParameters"],
        )
        if vis:
            visualizationCallback_kpr = eval.getVisualizationCallback(kpr)
            kpr.registerCallback(visualizationCallback_kpr)
        # setup result files
        trackingResult = setupTrackingResultTemplate()
        trackingResult["method"] = "kpr"
        eval.results["tracking"]["kpr"] = trackingResult
        for frame in framesToTrack:
            pointCloud = eval.getPointCloud(
                frame,
                dataSetPath,
            )
            Y = pointCloud[0]
            kpr.Y = Y
            # setup result callback
            logTargets = lambda: registrationResult["T"].append(kpr.T.copy())
            registrationResult = setupRegistrationResultTemplate()

            # run registration
            spr.register(checkConvergence=False, customCallback=logTargets)
            kpr.X = kpr.T.copy()
            # save results
            registrationResult["X"] = kpr.X.copy()
            registrationResult["Y"] = kpr.Y.copy()
            registrationResult["W"] = kpr.W.copy()
            registrationResult["G"] = kpr.G.copy()
            eval.results["tracking"]["spr"]["results"].append(registrationResult)

    if "krcpd" in registrationsToRun:
        krcpd = KinematicRegularizedCoherentPointDrift(
            Y=Y,
            qInit=qInit,
            model=KinematicsModelDart(bdloModel.skel.clone()),
            **eval.config["krcpdParameters"],
        )
        if vis:
            visualizationCallback_krcpd = eval.getVisualizationCallback(krcpd)
            krcpd.registerCallback(visualizationCallback_krcpd)
        # setup result files
        trackingResult = setupTrackingResultTemplate()
        trackingResult["method"] = "krcpd"
        eval.results["tracking"]["krcpd"] = trackingResult
        for frame in framesToTrack:
            pointCloud = eval.getPointCloud(
                frame,
                dataSetPath,
            )
            Y = pointCloud[0]
            krcpd.Y = Y
            # setup result callback
            logTargets = lambda: registrationResult["T"].append(krcpd.T.copy())
            registrationResult = setupRegistrationResultTemplate()
            # run registration
            krcpd.register(checkConvergence=False, customCallback=logTargets)
            krcpd.X = krcpd.T.copy()
            # save results
            registrationResult["X"] = krcpd.X.copy()
            registrationResult["Xreg"] = krcpd.Xreg.copy()
            registrationResult["Y"] = krcpd.Y.copy()
            registrationResult["W"] = krcpd.W.copy()
            registrationResult["G"] = krcpd.G.copy()
            eval.results["tracking"]["krcpd"]["results"].append(registrationResult)
        return


def evaluateResults(result):
    # compute tracking errors
    for key in result["tracking"]:
        trackingResults = result["tracking"][key]["results"]
        trackingErrors = []
        for trackingResult in trackingResults:
            T = trackingResult["T"][-1]
            Y = trackingResult["Y"]
            trackingError = np.sum(distance_matrix(T, Y))
            trackingErrors.append(trackingError)
        # add to error to result file
        result["tracking"][key]["trackingError"] = trackingErrors.copy()

    # plot tracking errors
    fig = plt.figure()
    ax = plt.axes()
    for key in result["tracking"]:
        trackingErrors = result["tracking"][key]["trackingError"]
        ax.plot(list(range(len(trackingErrors))), trackingErrors)
    plt.show(block=True)


if __name__ == "__main__":
    # determine initial and final frame for tracking
    initialFrame = eval.config["initialFrame"]
    if eval.config["finalFrame"] == -1:
        finalFrame = eval.getNumImageSetsInDataSet(dataSetPath)
    else:
        finalFrame = eval.config["finalFrame"]

    if runExperiment:
        if not loadInitialStateFromResult:
            # setup result file
            results = setupResultTemplate(dataSetPath)
            eval.results = results
            runModelGeneration(dataSetPath)
            runInitialLocalization(dataSetPath, initialFrame)
        else:
            results = eval.loadResults(resultFilePath)
            eval.results = results
        # run tracking evaluation
        trackingResult = runExperiment(dataSetPath, initialFrame, finalFrame)
        # save results
        if save:
            resultFilePath = eval.saveResults(
                folderPath=resultFolderPath,
                generateUniqueID=False,
                fileName=resultFileName,
            )

    # evaluate results
    if save:
        # load results from result file for evaluation
        results = eval.loadResults(resultFilePath)
        eval.results = results
        result = results
    else:
        result = eval.results
    evaluateResults(result)
