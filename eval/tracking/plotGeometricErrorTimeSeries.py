import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import pickle

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.visualization.plot2D import *
except:
    print("Imports for plotting script geometric error time series failed.")
    raise
controlOpt = {
    "resultsToLoad": [0],
    "highlightFrames": [[]],
    "save": True,
    "saveAsTikz": True,
    "showPlot": True,
    "saveFolder": "data/eval/tracking/geometricErrorTimeSeriesPlots",
    "saveName": "geometricErrorTimeSeries",
}
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


def createGeometricErrorTimeSeriesPlot(
    dataSetResult, lineColors=None, highlightFrames=None, highlightColor=[1, 0, 0]
):
    highlightFrames = [] if highlightFrames is None else highlightFrames
    fig = plt.figure()
    ax = plt.axes()
    geometricErrorLines = []
    for key in dataSetResult["trackingEvaluationResults"]:
        geometricErrors = dataSetResult["trackingEvaluationResults"][key][
            "geometricErrors"
        ]["mean"]
        (geometricErrorLine,) = ax.plot(
            list(range(len(geometricErrors))), geometricErrors
        )
        geometricErrorLine.set_label(key.upper())
        geometricErrorLines.append(geometricErrorLine)
    for highlightFrame in highlightFrames:
        ax.axvline(x=highlightFrame, color=highlightColor)

    # make legend
    ax.legend(loc="upper right")
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
        createGeometricErrorTimeSeriesPlot(result)
