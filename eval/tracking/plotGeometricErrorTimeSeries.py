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
    from src.visualization.plotUtils import *
    from src.visualization.colors import *
except:
    print("Imports for plotting script geometric error time series failed.")
    raise


global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [2],
    "save": True,
    "saveAs": "pdf",  # tikz, pdf, png
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/geometricErrorTimeSeries",
    "saveName": "geometricErrorTimeSeries",
    "methodsToEvaluate": ["cpd", "spr", "kpr"],  # "cpd", "spr", "kpr", "krcpd"
}
resultFileName = "result.pkl"

resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY",
    "data/eval/tracking/results/20230807_162939_ManipulationSequences_mountedWireHarness_partial",
    "data/eval/tracking/results/20230524_161235_ManipulationSequences_mountedWireHarness_arena",
]

styleOpt = {
    "plotAspectRatio": "default",  # default , golden
    "legende": True,
    "legendPosition": "upper left",
    "colorPalette": thesisColorPalettes["viridis"],
    "lineStyles": ["-", "-", "-"],  # line styles for CPD, SPR, KPR respectively
    "movingAverageWindowSize": 10,
    "movingAverageAlpha": 0.9,
    "movingAverageLineWidth": 1,
    "errorsAlpha": 0.3,
    "errorsLineStyle": "-",
    "errorsLineWidth": 3,
    "highlightColor": [1, 0, 0],
    "highlightFrames": [
        [400, 560, 630],
        [120, 150, 280],
        [55, 390, 465],
    ],  # list of list, each sublist for one data set
    "highlightLabels": [
        ["$t_1$", "$t_2$", "$t_3$"],
        ["$t_1$", "$t_2$", "$t_3$"],
        ["$t_1$", "$t_2$", "$t_3$"],
    ],  # list of list, each sublist for one data set
    "highlightColor": [1, 0, 0],
    "highlightTextColor": [0, 0, 0],
    "highlightAlpha": 0.5,
    "highlightLineStyles": [["--", "--", "--"], ["--", "--", "--"], ["--", "--", "--"]],
    "grid": True,
    "yAxisDescription": "length error in m",
    "cutOffThreshold": 5,  # leaves out last n values when plotting to avoid unreasonable dropoff from moving average
}

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


def loadResult(filePath):
    _, file_extension = os.path.splitext(filePath)
    if file_extension == ".pkl":
        with open(filePath, "rb") as f:
            result = pickle.load(f)
    return result


def createGeometricErrorTimeSeriesPlot(
    dataSetResult,
    methodsToEvaluate=None,
    lineStyles=None,
    highlightFrames=None,
    highlightColor=[1, 0, 0],
    highlightAlpha=1.0,
    highlightLineStyles=None,
    highlightLabels=None,
    highlightLabelFontSize=None,
    highlightTextColor=None,
    grid=True,
):
    lineStyles = ["-", "-", "-"] if lineStyles is None else lineStyles
    highlightFrames = [] if highlightFrames is None else highlightFrames
    highlightLineStyles = (
        ["-"] * len(highlightFrames)
        if highlightLineStyles is None
        else highlightLineStyles
    )
    highlightLabels = (
        [""] * len(highlightFrames) if highlightLabels is None else highlightLabels
    )
    highlightTextColor = [0, 0, 0] if highlightTextColor is None else highlightTextColor
    highlightLabelFontSize = (
        10 if highlightLabelFontSize is None else highlightLabelFontSize
    )

    # config
    trackingResults = dataSetResult["trackingResults"]
    methodsToEvaluate = (
        list(trackingResults.keys()) if methodsToEvaluate is None else methodsToEvaluate
    )
    grid = True if grid is None else grid

    if styleOpt["plotAspectRatio"] == "default":
        fig = plt.figure()
        ax = fig.add_subplot()
    elif styleOpt["plotAspectRatio"] == "golden":
        fig, ax = setupLatexPlot2D()
    else:
        raise NotImplementedError

    geometricErrorLines = []
    ymax = 0  # Initialize a variable to track the maximum y value

    for i, method in enumerate(methodsToEvaluate):
        color = styleOpt["colorPalette"].to_rgba(i / (len(methodsToEvaluate) - 1))[:3]
        geometricErrorResults = eval.calculateGeometricErrors(trackingResults[method])
        geometricErrors = geometricErrorResults["lengthError"]
        geometricErrorsMovingAvg = np.convolve(
            geometricErrors,
            np.ones(styleOpt["movingAverageWindowSize"])
            / styleOpt["movingAverageWindowSize"],
            mode="same",
        )
        (geometricErrorLine,) = ax.plot(
            list(range(len(geometricErrors[: -styleOpt["cutOffThreshold"]]))),
            geometricErrors[: -styleOpt["cutOffThreshold"]],
            color=color,
            linestyle=styleOpt["errorsLineStyle"],
            linewidth=styleOpt["errorsLineWidth"],
            alpha=styleOpt["errorsAlpha"],
        )
        (geometricErrorLine,) = ax.plot(
            list(range(len(geometricErrors[: -styleOpt["cutOffThreshold"]]))),
            geometricErrorsMovingAvg[: -styleOpt["cutOffThreshold"]],
            color=color,
            linestyle=lineStyles[i],
            linewidth=styleOpt["movingAverageLineWidth"],
            alpha=styleOpt["movingAverageAlpha"],
        )
        geometricErrorLine.set_label(method.upper())
        geometricErrorLines.append(geometricErrorLine)
        ymax = max(ymax, max(geometricErrors))  # Update ymax

    for i, highlightFrame in enumerate(highlightFrames):
        linestyle = highlightLineStyles[i] if i < len(highlightLineStyles) else "-"
        label = highlightLabels[i] if i < len(highlightLabels) else ""

        # Plot vertical line
        ax.axvline(
            x=highlightFrame,
            color=highlightColor,
            alpha=highlightAlpha,
            linestyle=linestyle,
        )

        # Add label above the vertical line at a slightly higher y position
        if label:
            ax.text(
                highlightFrame,
                ymax * 1.05,
                label,
                rotation=0,
                color="black",
                ha="center",
                va="bottom",
                fontsize=highlightLabelFontSize,
            )

    # customize layout
    if grid:
        plt.grid(True)
    if styleOpt["legende"]:
        # make legend
        ax.legend(loc=styleOpt["legendPosition"])
    plt.xlabel("frames")
    plt.ylabel(styleOpt["yAxisDescription"])

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # saving
    if controlOpt["save"]:
        # make folder for dataSet
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        saveFolderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
        if not os.path.exists(saveFolderPath):
            os.makedirs(saveFolderPath)
        fileName = controlOpt["saveName"]
        savePath = os.path.join(saveFolderPath, fileName)
        if controlOpt["saveAs"] == "png":
            # save as png
            plt.savefig(savePath)
        # save as pdf
        if controlOpt["saveAs"] == "pdf":
            plt.savefig(savePath + ".pdf", bbox_inches="tight")
        # save as tixfigure
        if controlOpt["saveAs"] == "tikz":
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
    for i, result in enumerate(results):
        createGeometricErrorTimeSeriesPlot(
            result,
            methodsToEvaluate=controlOpt["methodsToEvaluate"],
            highlightFrames=styleOpt["highlightFrames"][controlOpt["resultsToLoad"][i]],
            highlightColor=styleOpt["highlightColor"],
            lineStyles=styleOpt["lineStyles"],
            highlightLabels=styleOpt["highlightLabels"][controlOpt["resultsToLoad"][i]],
            highlightAlpha=styleOpt["highlightAlpha"],
            highlightLineStyles=styleOpt["highlightLineStyles"][
                controlOpt["resultsToLoad"][i]
            ],
            highlightLabelFontSize=latexFontSize_in_pt,
            grid=styleOpt["grid"],
        )
