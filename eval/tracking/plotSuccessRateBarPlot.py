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
    from src.visualization.colors import *
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
    "saveAs": "pdf",  # "pdf" , "png", "tikz"
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/successRateBarPlot",
    "saveName": "successRateBarPlot",
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
}

styleOpt = {
    "legende": False,
    "colorPalette": thesisColorPalettes["viridis"],
    "grid": True,
}

# successfullyTrackedFrames = {
#     "20230524_171237_ManipulationSequences_mountedWireHarness_modelY": {
#         "cpd": 397,
#         "spr": 550,
#         "kpr": 695,
#         "krcpd": 695,
#     },
#     "20230807_162939_ManipulationSequences_mountedWireHarness_partial": {
#         "cpd": 120,
#         "spr": 120,
#         "kpr": 315,
#         "krcpd": 315,
#     },
#     "20230524_161235_ManipulationSequences_mountedWireHarness_arena": {
#         "cpd": 55,
#         "spr": 380,
#         "kpr": 497,
#         "krcpd": 497,
#     },
# }

# figure font configuration
latexFontSize_in_pt = 16
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFontSize_in_pt,
    "xtick.labelsize": latexFontSize_in_pt,
    "ytick.labelsize": latexFontSize_in_pt,
}
plt.rcParams.update(tex_fonts)

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
    spacingFactor=1.0,
    legend=False,
):
    trackingResults = dataSetResult["trackingResults"]
    methodsToEvaluate = (
        list(trackingResults.keys()) if methodsToEvaluate is None else methodsToEvaluate
    )
    dataSetName = dataSetResult["dataSetPath"].split("/")[-2]
    # plot error bars
    XTicks = np.arange(len(methodsToEvaluate)) * spacingFactor

    # open plot
    # fig, ax = setupLatexPlot2D()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    ax.grid(styleOpt["grid"])

    plotColors = []
    for i, (x, method) in enumerate(zip(XTicks, methodsToEvaluate)):
        bar_color = styleOpt["colorPalette"].to_rgba(i / (len(methodsToEvaluate) - 1))[
            :3
        ]
        # barValue = successfullyTrackedFrames[dataSetName][method]
        successRateResults = eval.calculateSuccessRate(trackingResults[method])
        barValue = successRateResults["numSuccessfullyTrackedFrames"]
        successRate = successRateResults["successRate"] * 100
        absNumFrames = len(trackingResults[method]["frames"])
        bottomValue = barValue
        successBar = ax.bar(x, barValue, barWidth, color=bar_color)
        # plot unsuccessful bar
        unsuccessBar = ax.bar(
            x,
            absNumFrames - barValue,
            barWidth,
            bottom=bottomValue,
            color="gray",
        )
        plotColors.append(successBar.patches[0].get_facecolor())
        for unsuccessRect, successRect in zip(unsuccessBar, successBar):
            height = unsuccessRect.get_height() + successRect.get_height()
            plt.text(
                successRect.get_x() + successRect.get_width() / 2.0,
                height,
                "{:.1f}".format(successRate) + "\%",
                ha="center",
                va="bottom",
            )

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
    plt.ylabel("successfully tracked frames")
    plt.xticks(XTicks, [x.upper() for x in methodsToEvaluate])
    plt.ylim([0, int(absNumFrames * 1.2)])
    if legend:
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
    # Adjust layout to prevent clipping
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
        if controlOpt["saveAs"] == "png":
            plt.savefig(savePath)
        # save as tixfigure
        if controlOpt["saveAs"] == "tikz":
            tikzplotlib.save(savePath + ".tex")
        if controlOpt["saveAs"] == "pdf":
            plt.savefig(savePath + ".pdf", bbox_inches="tight")
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
                dataSetResult,
                methodsToEvaluate=controlOpt["methodsToEvaluate"],
                legend=styleOpt["legende"],
            )
