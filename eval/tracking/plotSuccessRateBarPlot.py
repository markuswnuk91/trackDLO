import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import pickle
from warnings import warn
from matplotlib.patches import Patch

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot2D import *
except:
    print("Imports for plotting reprojection errors as boxplot failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [0, 1, 2],
    "createPlot": True,
    "printNumTrackedFrames": False,
    "save": True,
    "saveAsTikz": True,
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/successRateBarPlot",
    "saveName": "successRateBarPlot",
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
}

successfullyTrackedFrames = {
    "20230524_171237_ManipulationSequences_mountedWireHarness_modelY": {
        "cpd": 397,
        "spr": 550,
        "kpr": 695,
        "krcpd": 695,
    },
    "20230807_162939_ManipulationSequences_mountedWireHarness_partial": {
        "cpd": 120,
        "spr": 120,
        "kpr": 315,
        "krcpd": 315,
    },
    "20230524_161235_ManipulationSequences_mountedWireHarness_arena": {
        "cpd": 55,
        "spr": 380,
        "kpr": 497,
        "krcpd": 497,
    },
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


def printNumTrackedFrames(dataSetResult):
    availableMethods = list(dataSetResult["trackingResults"].keys())
    dataSetName = dataSetResult["dataSetPath"].split("/")[-2]
    print(
        "Dataset: {} has {} frames.".format(
            dataSetName,
            len(dataSetResult["trackingResults"][availableMethods[0]]["frames"]),
        )
    )


def createSuccessRateBarPlot(
    dataSetResult,
    methodsToEvaluate=None,
    barWidth=0.5,
):
    trackingResults = dataSetResult["trackingResults"]
    methodsToEvaluate = (
        list(trackingResults.keys()) if methodsToEvaluate is None else methodsToEvaluate
    )
    dataSetName = dataSetResult["dataSetPath"].split("/")[-2]
    # plot error bars
    XTicks = np.arange(len(methodsToEvaluate)) * spacingFactor

    plotColors = []
    for x, method in zip(XTicks, methodsToEvaluate):
        barValue = successfullyTrackedFrames[dataSetName][method]
        absNumFrames = len(trackingResults[method]["frames"])
        bottomValue = barValue
        successBar = plt.bar(x, barValue, barWidth)
        plt.bar(
            x,
            absNumFrames - barValue,
            barWidth,
            bottom=bottomValue,
            color="gray",
        )
        plotColors.append(successBar.patches[0].get_facecolor())

    # barValues = []
    # absNumFrames = []
    # bottom = np.zeros(len(methodsToEvaluate))
    # for method in methodsToEvaluate:
    #     barValues.append(successfullyTrackedFrames[dataSetName][method])
    #     absNumFrames.append(len(trackingResults[method]["frames"]))
    # barValues = np.array(barValues)
    # absNumFrames = np.array(absNumFrames)

    # plt.bar(XTicks, barValues, barWidth, bottom=bottom, label="successful")
    # bottom += barValues
    # plt.bar(
    #     XTicks,
    #     absNumFrames - barValues,
    #     barWidth,
    #     bottom=barValues,
    #     color="gray",
    #     label="not successful",
    # )

    # Set the title, labels, and a legend
    plt.xlabel("methods")
    plt.ylabel("Successfully tracked frames")
    plt.xticks(XTicks, [x.upper() for x in methodsToEvaluate])
    plt.ylim([0, int(absNumFrames * 1.2)])
    plt.legend()
    pa1 = Patch(facecolor=plotColors[0], edgecolor="black")
    pa2 = Patch(facecolor=plotColors[1], edgecolor="black")
    pa3 = Patch(facecolor=plotColors[2], edgecolor="black")
    pb1 = Patch(facecolor="gray", edgecolor="gray")
    pb2 = Patch(facecolor="gray", edgecolor="gray")
    pb3 = Patch(facecolor="gray", edgecolor="gray")
    plt.legend(
        handles=[pa1, pb1, pa2, pb2, pa3, pb3],
        labels=["", "", "", "", "successful", "not successful"],
        ncol=3,
        handletextpad=0.5,
        handlelength=1.0,
        columnspacing=-0.5,
    )
    plt.tight_layout()

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
    return


if __name__ == "__main__":
    # load all results
    dataSetResults = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        dataSetResults.append(result)
    for dataSetResult in dataSetResults:
        if controlOpt["printNumTrackedFrames"]:
            printNumTrackedFrames(dataSetResult)
        if controlOpt["createPlot"]:
            # create plot
            createSuccessRateBarPlot(
                dataSetResult, methodsToEvaluate=controlOpt["methodsToEvaluate"]
            )
