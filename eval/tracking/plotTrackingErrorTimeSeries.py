import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import pickle

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot2D import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [0, 1, 2],
    "highlightFrames": [[], [], []],
    "save": True,
    "saveAsTikz": True,
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/trackingErrorTimeSeries",
    "saveName": "trackingErrorTimeSeries",
    "methodsToEvaluate": ["cpd", "spr", "kpr"],  # "cpd", "spr", "kpr", "krcpd"
}
styleOpt = {"legende": True}
resultFileName = "result.pkl"

resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY",
    "data/eval/tracking/results/20230807_162939_ManipulationSequences_mountedWireHarness_partial",
    "data/eval/tracking/results/20230524_161235_ManipulationSequences_mountedWireHarness_arena",
]


def loadResult(filePath):
    _, file_extension = os.path.splitext(filePath)
    if file_extension == ".pkl":
        with open(filePath, "rb") as f:
            result = pickle.load(f)
    return result


def createTrackingErrorTimeSeriesPlot(
    dataSetResult,
    methodsToEvaluate=None,
    lineColors=None,
    highlightFrames=None,
    highlightColor=[1, 0, 0],
):
    """creates for a data set a folder with a tracking error plot over all methods, and images for each method for certain frames which can be custumized and are indicated as vertial lines in the plot"""

    trackingResults = dataSetResult["trackingResults"]

    methodsToEvaluate = (
        list(trackingResults.keys()) if methodsToEvaluate is None else methodsToEvaluate
    )

    highlightFrames = [] if highlightFrames is None else highlightFrames
    fig = plt.figure()
    ax = plt.axes()
    trackingErrorLines = []
    for method in methodsToEvaluate:
        trackingErrors = eval.calculateTrackingErrors(trackingResults[method])
        (trackingErrorLine,) = ax.plot(list(range(len(trackingErrors))), trackingErrors)
        trackingErrorLine.set_label(method.upper())
        trackingErrorLines.append(trackingErrorLine)
    for highlightFrame in highlightFrames:
        ax.axvline(x=highlightFrame, color=highlightColor)

    if styleOpt["legende"]:
        # make legend
        ax.legend(loc="upper right")
    plt.xlabel("frames")
    plt.ylabel("tracking error in m")
    if controlOpt["save"]:
        # make folder for dataSet
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        saveFolderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
        if not os.path.exists(saveFolderPath):
            os.makedirs(saveFolderPath)
        fileName = controlOpt["saveName"]
        savePath = os.path.join(saveFolderPath, fileName)
        # save as png
        plt.savefig(savePath)
        # save as tixfigure
        if controlOpt["saveAsTikz"]:
            tikzplotlib.save(savePath + ".tex")
    if controlOpt["showPlot"]:
        plt.show(block=True)


if __name__ == "__main__":
    # load all results
    results = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        results.append(result)
    # create plot
    for result in results:
        createTrackingErrorTimeSeriesPlot(
            dataSetResult=result, methodsToEvaluate=controlOpt["methodsToEvaluate"]
        )
